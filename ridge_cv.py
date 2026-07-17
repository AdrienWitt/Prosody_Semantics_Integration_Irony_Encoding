import os
import numpy as np
import logging
from analysis_helpers import mult_diag, counter
from dataset import FoldPreprocessor
import random
import itertools as itools
from sklearn.model_selection import GroupKFold
from joblib import Parallel, delayed
import sys
import pandas as pd


ridge_logger = logging.getLogger("ridge_corr")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
backend = 'multiprocessing'
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

def zs(v):
    std = v.std(0)
    std[std == 0] = 1  # avoid divide-by-zero; constant columns stay at zero
    return (v - v.mean(0)) / std

def ridge_cv(stim_df, resp, alphas, participant_ids,
             col_groups, use_pca=False, pca_threshold=0.95,
             n_lopo=50, n_splits=50,
             corrmin=0, singcutoff=1e-10, normalpha=False, use_corr=True,
             return_wt=False, normalize_stim=False, normalize_resp=True,
             n_jobs=1, with_replacement=False, optimize_alpha=True,
             valphas=None, logger=None):
    """
    Performs ridge regression with K-group cross-validation and/or LOPO alpha optimization.
    Combines group-based CV and alpha optimization into a single function with efficient group splitting.
    If n_groups equals the number of unique participants, it performs leave-one-participant-out CV.

    Parameters
    ----------
    stim_df : pandas.DataFrame, shape (T, N+P-1[+K-1])
        Stimuli with T time points, N features, P-1 participant covariates, K-1 task covariates.
    resp : array_like, shape (T, M)
        fMRI responses with T time points and M voxels.
    alphas : list or array_like, shape (A,)
        Ridge parameters to test (e.g., np.logspace(0, 3, 20)) when optimize_alpha=True.
    participant_ids : array_like, shape (T,)
        Participant IDs for each time point.
    n_lopo : int, default 50
        Number of LOPO iterations for alpha optimization (if optimize_alpha=True).
    n_groups : int, default 50
        Number of groups for K-group CV (if equal to number of participants, performs LOPO CV).
    corrmin : float, default 0.1
        Minimum correlation for logging.
    singcutoff : float, default 1e-10
        Singular value cutoff for SVD.
    normalpha : boolean, default False
        Normalize alphas by largest singular value.
    use_corr : boolean, default True
        Use correlation (True) or R-squared (False).
    return_wt : boolean, default False
        Return regression weights.
    normalize_stim : boolean, default False
        Z-score stimuli (False if pre-normalized).
    normalize_resp : boolean, default True
        Z-score responses (True for per-task-normalized data).
    n_jobs : int, default 1
        Number of parallel jobs for CV folds or LOPO iterations (-1 for all cores).
    with_replacement : boolean, default False
        Sample participants with replacement in LOPO alpha optimization (if optimize_alpha=True).
    optimize_alpha : boolean, default True
        Optimize alpha using LOPO iterations. If False, valphas must be provided.
    valphas : array_like, shape (M,), default None
        Precomputed optimal alpha per voxel, required if optimize_alpha=False.
    logger : logging.Logger, default None
        Logger for tracking progress.

    Returns
    -------
    wt : array_like, shape (N+P-1[+K-1], M)
        Regression weights (empty if return_wt=False).
    corrs : array_like, shape (M,)
        Average correlation across CV folds (empty if n_groups=0).
    valphas : array_like, shape (M,)
        Optimal alpha per voxel (either optimized or provided).
    fold_corrs : array_like, shape (M, n_groups)
        Correlations per voxel per CV fold (empty if n_groups=0).
    lopo_corrs : array_like, shape (A, M, n_lopo)
        Correlations for each alpha, voxel, and LOPO iteration (empty if optimize_alpha=False).
    """
    if len(stim_df) != resp.shape[0]:
        raise ValueError("stim_df and resp must have same number of time points.")

    resp = zs(resp) if normalize_resp else resp
    
    if participant_ids is None:
        raise ValueError("participant_ids required for group CV and LOPO alpha optimization.")
    
    unique_participants = np.unique(participant_ids)
    n_participants = len(unique_participants)
    
    # Handle alpha optimization
    lopo_corrs = None
    if optimize_alpha:
        logger.info("Starting LOPO alpha optimization with %d iterations...", n_lopo) if logger else None
        if n_lopo > n_participants and not with_replacement:
            raise ValueError(f"n_lopo ({n_lopo}) cannot exceed number of participants ({n_participants}) without replacement.")

        np.random.seed(42)
        participant_choices = (np.random.choice(unique_participants, size=n_lopo, replace=True)
                              if with_replacement else
                              np.random.permutation(unique_participants)[:min(n_lopo, n_participants)])

        def _lopo_iter(val_participant, iteration, total):
            logger.info(f"LOPO iteration {iteration+1}/{total} for participant {val_participant}...") if logger else None
            heldinds = participant_ids == val_participant
            notheldinds = ~heldinds
            prep = FoldPreprocessor(col_groups, use_pca, pca_threshold)
            RRstim = prep.fit_transform(stim_df.iloc[notheldinds])
            PRstim = prep.transform(stim_df.iloc[heldinds])
            RRresp, PRresp = resp[notheldinds, :], resp[heldinds, :]
            return ridge_corr(RRstim, PRstim, RRresp, PRresp, alphas,
                             corrmin=corrmin, singcutoff=singcutoff,
                             normalpha=normalpha, use_corr=use_corr, logger=logger)

        lopo_results = Parallel(n_jobs=n_jobs)(
            delayed(_lopo_iter)(val_participant, i, n_lopo)
            for i, val_participant in enumerate(participant_choices)
        )

        lopo_corrs = np.dstack(lopo_results) if n_lopo > 0 else None
        if lopo_corrs is not None:
            mean_lopo_corrs = lopo_corrs.mean(2)
            bestalphainds = np.argmax(mean_lopo_corrs, 0)
            valphas = alphas[bestalphainds]
            for ua in np.unique(valphas):
                sel_vox = np.nonzero(valphas == ua)[0]
                mean_corr = np.mean(mean_lopo_corrs[bestalphainds[sel_vox], sel_vox]) if len(sel_vox) > 0 else 0
                logger.info("Alpha=%0.3f selected for %d voxels (mean corr=%0.5f)", ua, len(sel_vox), mean_corr) if logger else None
    else:
        if valphas is None:
            raise ValueError("valphas must be provided when optimize_alpha=False.")
        if not isinstance(valphas, np.ndarray) or valphas.shape != (resp.shape[1],):
            raise ValueError(f"valphas must be a numpy array of shape ({resp.shape[1]},).")
        logger.info("Using provided valphas for cross-validation...") if logger else None
    
    # Perform K-group cross-validation
    fold_corrs = []
    if n_splits > 0:
        logger.info("Performing %d-group cross-validation...", n_splits) if logger else None
        n_splits = min(n_splits, n_participants)  # Ensure n_groups does not exceed n_participants
        gkf = GroupKFold(n_splits=n_splits)
        
        def _cv_iter(fold_idx, train_idx, test_idx):
            logger.info(f"Processing CV fold {fold_idx+1}/{n_splits}...") if logger else None
            prep = FoldPreprocessor(col_groups, use_pca, pca_threshold)
            Rstim = prep.fit_transform(stim_df.iloc[train_idx])
            Pstim = prep.transform(stim_df.iloc[test_idx])
            Rresp, Presp = resp[train_idx], resp[test_idx]
            return ridge_corr_pred(Rstim, Pstim, Rresp, Presp, valphas,
                                  normalpha=normalpha, singcutoff=singcutoff,
                                  use_corr=use_corr, logger=logger)

        dummy_X = np.arange(len(participant_ids)).reshape(-1, 1)
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(_cv_iter)(fold_idx, train_idx, test_idx)
            for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(dummy_X, groups=participant_ids))
        )
        
        fold_corrs = np.stack(fold_results, axis=1) if fold_results else np.array([])
        if fold_corrs.size > 0:
            corrs = np.mean(fold_corrs, axis=1)
            logger.info("Completed CV: mean correlation across voxels=%0.5f, max=%0.5f",
                        np.mean(corrs), np.max(corrs)) if logger else None
        else:
            corrs = np.array([])
    else:
        corrs = np.array([])
        logger.info("Skipping CV as n_groups=0.") if logger else None
    
    # Compute weights if requested — fit preprocessor on full dataset (weights only, not used for evaluation)
    wt = []
    if return_wt:
        logger.info("Computing weights on full dataset...") if logger else None
        prep_full = FoldPreprocessor(col_groups, use_pca, pca_threshold)
        stim_full = prep_full.fit_transform(stim_df)
        wt = ridge(stim_full, resp, valphas, singcutoff=singcutoff, normalpha=normalpha, logger=logger)
        logger.info("Weights computed for %d features and %d voxels.", wt.shape[0], wt.shape[1]) if logger else None
    
    return wt, corrs, valphas, fold_corrs, lopo_corrs




def ridge(stim, resp, alpha, singcutoff=1e-10, normalpha=False, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [stim] that approximates
    [resp]. The regularization parameter is [alpha].

    Parameters
    ----------
    stim : array_like, shape (T, N)
        Stimuli with T time points and N features.
    resp : array_like, shape (T, M)
        Responses with T time points and M separate responses.
    alpha : float or array_like, shape (M,)
        Regularization parameter. Can be given as a single value (which is applied to
        all M responses) or separate values for each response.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value of stim. Good for
        comparing models with different numbers of parameters.

    Returns
    -------
    wt : array_like, shape (N, M)
        Linear regression weights.
    """
    try:
        U,S,Vh = np.linalg.svd(stim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(stim, full_matrices=False)

    UR = np.dot(U.T, np.nan_to_num(resp))
    
    # Expand alpha to a collection if it's just a single value
    if isinstance(alpha, (float,int)):
        alpha = np.ones(resp.shape[1]) * alpha
    
    # Normalize alpha by the LSV norm
    norm = S[0]
    if normalpha:
        nalphas = alpha * norm
    else:
        nalphas = alpha

    # Compute weights for each alpha
    ualphas = np.unique(nalphas)
    wt = np.zeros((stim.shape[1], resp.shape[1]))
    for ua in ualphas:
        selvox = np.nonzero(nalphas==ua)[0]
        #awt = reduce(np.dot, [Vh.T, np.diag(S/(S**2+ua**2)), UR[:,selvox]])
        awt = Vh.T.dot(np.diag(S/(S**2+ua**2))).dot(UR[:,selvox])
        wt[:,selvox] = awt

    return wt
    
def ridge_corr_pred(Rstim, Pstim, Rresp, Presp, valphas, normalpha=False,
                    singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. Returns the correlation 
    between predicted and actual [Presp], without ever computing the regression weights.
    This function assumes that each voxel is assigned a separate alpha in [valphas].

    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    valphas : list or array_like, shape (M,)
        Ridge parameter for each voxel.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    corr : array_like, shape (M,)
        The correlation between each predicted response and each column of Presp.
    
    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    ## Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: %0.03f"%norm)
    if normalpha:
        nalphas = valphas * norm
    else:
        nalphas = valphas

    ## Precompute some products for speed
    UR = np.dot(U.T, Rresp) ## Precompute this matrix product for speed
    PVh = np.dot(Pstim, Vh.T) ## Precompute this matrix product for speed
    
    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(Presp)
    #Prespvar = Presp.var(0)
    Prespvar_actual = Presp.var(0)
    Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (Prespvar_actual - Prespvar).mean())

    ualphas = np.unique(nalphas)
    corr = np.zeros((Rresp.shape[1],))
    for ua in ualphas:
        selvox = np.nonzero(nalphas==ua)[0]
        alpha_pred = PVh.dot(np.diag(S/(S**2+ua**2))).dot(UR[:,selvox])

        if use_corr:
            corr[selvox] = (zPresp[:,selvox] * zs(alpha_pred)).mean(0)
        else:
            resvar = (Presp[:,selvox] - alpha_pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            corr[selvox] = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)

    return corr


def ridge_corr(Rstim, Pstim, Rresp, Presp, alphas, normalpha=False, corrmin=0.2,
               singcutoff=1e-10, use_corr=True, logger=ridge_logger):
    """Uses ridge regression to find a linear transformation of [Rstim] that approximates [Rresp],
    then tests by comparing the transformation of [Pstim] to [Presp]. This procedure is repeated
    for each regularization parameter alpha in [alphas]. The correlation between each prediction and
    each response for each alpha is returned. The regression weights are NOT returned, because
    computing the correlations without computing regression weights is much, MUCH faster.

    Parameters
    ----------
    Rstim : array_like, shape (TR, N)
        Training stimuli with TR time points and N features. Each feature should be Z-scored across time.
    Pstim : array_like, shape (TP, N)
        Test stimuli with TP time points and N features. Each feature should be Z-scored across time.
    Rresp : array_like, shape (TR, M)
        Training responses with TR time points and M responses (voxels, neurons, what-have-you).
        Each response should be Z-scored across time.
    Presp : array_like, shape (TP, M)
        Test responses with TP time points and M responses.
    alphas : list or array_like, shape (A,)
        Ridge parameters to be tested. Should probably be log-spaced. np.logspace(0, 3, 20) works well.
    normalpha : boolean
        Whether ridge parameters should be normalized by the largest singular value (LSV) norm of
        Rstim. Good for comparing models with different numbers of parameters.
    corrmin : float in [0..1]
        Purely for display purposes. After each alpha is tested, the number of responses with correlation
        greater than corrmin minus the number of responses with correlation less than negative corrmin
        will be printed. For long-running regressions this vague metric of non-centered skewness can
        give you a rough sense of how well the model is working before it's done.
    singcutoff : float
        The first step in ridge regression is computing the singular value decomposition (SVD) of the
        stimulus Rstim. If Rstim is not full rank, some singular values will be approximately equal
        to zero and the corresponding singular vectors will be noise. These singular values/vectors
        should be removed both for speed (the fewer multiplications the better!) and accuracy. Any
        singular values less than singcutoff will be removed.
    use_corr : boolean
        If True, this function will use correlation as its metric of model fit. If False, this function
        will instead use variance explained (R-squared) as its metric of model fit. For ridge regression
        this can make a big difference -- highly regularized solutions will have very small norms and
        will thus explain very little variance while still leading to high correlations, as correlation
        is scale-free while R**2 is not.

    Returns
    -------
    Rcorrs : array_like, shape (A, M)
        The correlation between each predicted response and each column of Presp for each alpha.
    
    """
    ## Calculate SVD of stimulus matrix
    logger.info("Doing SVD...")
    try:
        U,S,Vh = np.linalg.svd(Rstim, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.info("NORMAL SVD FAILED, trying more robust dgesvd..")
        from text.regression.svd_dgesvd import svd_dgesvd
        U,S,Vh = svd_dgesvd(Rstim, full_matrices=False)

    ## Truncate tiny singular values for speed
    origsize = S.shape[0]
    ngoodS = np.sum(S > singcutoff)
    nbad = origsize-ngoodS
    U = U[:,:ngoodS]
    S = S[:ngoodS]
    Vh = Vh[:ngoodS]
    logger.info("Dropped %d tiny singular values.. (U is now %s)"%(nbad, str(U.shape)))

    ## Normalize alpha by the LSV norm
    norm = S[0]
    logger.info("Training stimulus has LSV norm: %0.03f"%norm)
    if normalpha:
        nalphas = alphas * norm
    else:
        nalphas = alphas

    ## Precompute some products for speed
    UR = np.dot(U.T, Rresp) ## Precompute this matrix product for speed
    PVh = np.dot(Pstim, Vh.T) ## Precompute this matrix product for speed
    
    #Prespnorms = np.apply_along_axis(np.linalg.norm, 0, Presp) ## Precompute test response norms
    zPresp = zs(Presp)
    #Prespvar = Presp.var(0)
    Prespvar_actual = Presp.var(0)
    Prespvar = (np.ones_like(Prespvar_actual) + Prespvar_actual) / 2.0
    logger.info("Average difference between actual & assumed Prespvar: %0.3f" % (Prespvar_actual - Prespvar).mean())
    Rcorrs = [] ## Holds training correlations for each alpha
    for na, a in zip(nalphas, alphas):
        #D = np.diag(S/(S**2+a**2)) ## Reweight singular vectors by the ridge parameter 
        D = S / (S ** 2 + na ** 2) ## Reweight singular vectors by the (normalized?) ridge parameter
        
        pred = np.dot(mult_diag(D, PVh, left=False), UR) ## Best (1.75 seconds to prediction in test)
        # pred = np.dot(mult_diag(D, np.dot(Pstim, Vh.T), left=False), UR) ## Better (2.0 seconds to prediction in test)
        
        # pvhd = reduce(np.dot, [Pstim, Vh.T, D]) ## Pretty good (2.4 seconds to prediction in test)
        # pred = np.dot(pvhd, UR)
        
        # wt = reduce(np.dot, [Vh.T, D, UR]).astype(dtype) ## Bad (14.2 seconds to prediction in test)
        # wt = reduce(np.dot, [Vh.T, D, U.T, Rresp]).astype(dtype) ## Worst
        # pred = np.dot(Pstim, wt) ## Predict test responses

        if use_corr:
            #prednorms = np.apply_along_axis(np.linalg.norm, 0, pred) ## Compute predicted test response norms
            #Rcorr = np.array([np.corrcoef(Presp[:,ii], pred[:,ii].ravel())[0,1] for ii in range(Presp.shape[1])]) ## Slowly compute correlations
            #Rcorr = np.array(np.sum(np.multiply(Presp, pred), 0)).squeeze()/(prednorms*Prespnorms) ## Efficiently compute correlations
            Rcorr = (zPresp * zs(pred)).mean(0)
        else:
            ## Compute variance explained
            resvar = (Presp - pred).var(0)
            Rsq = 1 - (resvar / Prespvar)
            Rcorr = np.sqrt(np.abs(Rsq)) * np.sign(Rsq)
            
        Rcorr[np.isnan(Rcorr)] = 0
        Rcorrs.append(Rcorr)
        
        log_template = "Training: alpha=%0.3f, mean corr=%0.5f, max corr=%0.5f, over-under(%0.2f)=%d"
        log_msg = log_template % (a,
                                  np.mean(Rcorr),
                                  np.max(Rcorr),
                                  corrmin,
                                  (Rcorr>corrmin).sum()-(-Rcorr>corrmin).sum())
        logger.info(log_msg)
    
    return Rcorrs



