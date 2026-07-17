import os
import time
import argparse
import numpy as np
import analysis_helpers
import nibabel as nib
from nilearn.image import resample_to_img
import logging
from ridge_cv import ridge_cv
from nilearn import datasets, image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
ridge_logger = logging.getLogger("ridge_corr")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run fMRI Ridge Regression analysis.")

    # Dataset-related arguments
    dataset_group = parser.add_argument_group("Dataset Arguments")
    dataset_group.add_argument("--use_base_features", action="store_true", 
                               help="Include base features in dataset (default: False).")
    dataset_group.add_argument("--use_text", action="store_true", 
                               help="Include text in dataset (default: False).")
    dataset_group.add_argument("--use_audio", action="store_true",
                               help="Include audio in dataset (default: False).")
    dataset_group.add_argument("--use_interaction", action="store_true",
                               help="Add the prosody x semantic interaction term (needs --use_text and --use_audio).")
    dataset_group.add_argument("--text_embedding_type", type=str, default="cross_attention",
                               choices=["cross_attention", "statement_only"],
                               help="cross_attention = statement re-weighted by context; "
                                    "statement_only = statement without context.")
    dataset_group.add_argument("--use_pca", action="store_true",
                               help="Use PCA for embeddings with a certain amount of explained variance directly in the dataset method (default: False).")
    dataset_group.add_argument("--pca_threshold", type=float, default=0.60,
                               help="Explained variance threshold for PCA (default: 0.60).")
    dataset_group.add_argument("--include_tasks", type=str, nargs='+', default=["sarcasm", "irony", "prosody", "semantic", "tom"],
                               help="List of tasks to include (default: all available tasks).")
    # Analysis-related arguments
    analysis_group = parser.add_argument_group("Analysis Arguments")
    analysis_group.add_argument("--num_jobs", type=int, default=-1,
                               help="Number of parallel jobs for voxel processing (default: -1 for all cores).")
    analysis_group.add_argument("--optimize_alpha", action="store_true",
                               help="Optimize alpha values using LOPO iterations (default: False, use precomputed valphas if provided).")
    analysis_group.add_argument("--alpha_min", type=float, default=-2,
                               help="Minimum exponent for alpha values in logspace (default: -2).")
    analysis_group.add_argument("--alpha_max", type=float, default=3,
                               help="Maximum exponent for alpha values in logspace (default: 3).")
    analysis_group.add_argument("--num_alphas", type=int, default=10,
                               help="Number of alpha values to test in logspace (default: 10).")
    analysis_group.add_argument("--n_lopo", type=int, default=None,
                               help="Number of LOPO iterations for alpha optimization (default: number of participants).")
    analysis_group.add_argument("--corrmin", type=float, default=0.0,
                               help="Minimum correlation threshold for fMRI correlations (default: 0.0).")
    analysis_group.add_argument("--n_splits", type=int, default=None,
                               help="Number of splits for cross-validation (default: number of participants for LOO CV).")
    analysis_group.add_argument("--normalpha", action="store_true",
                               help="Normalize alpha values for reuse across models (default: False).")
    analysis_group.add_argument("--use_corr", action="store_true", default = True,
                               help="Use correlation as the evaluation metric (default: False).")
    analysis_group.add_argument("--return_wt", action="store_true",
                               help="Return weight maps from ridge regression (default: False).")
    analysis_group.add_argument("--normalize_resp", action="store_true", default=True,
                               help="Normalize response (fMRI) data before regression (default: True).")
    analysis_group.add_argument("--with_replacement", action="store_true",
                               help="Sample participants with replacement during LOPO alpha optimization (default: False).")
    analysis_group.add_argument("--results_dir", type=str, default="results",
                               help="Custom directory to save results (default: uses paths from analysis_helpers).")


    return parser.parse_args()

def main():
    start_time = time.time()  # Start timing

    args = parse_arguments()

    # The interaction term only exists when both modalities are present.
    interaction_active = args.use_interaction and args.use_text and args.use_audio
    if args.use_interaction and not interaction_active:
        print("WARNING: --use_interaction requires both --use_text and --use_audio; ignored.")
    # Suffix the folder with '_interaction' so it can't overwrite a no-interaction run.
    if interaction_active and not args.results_dir.rstrip("/").endswith("_interaction"):
        args.results_dir = args.results_dir.rstrip("/") + "_interaction"

    # Print settings, including new arguments
    print(f"Running with settings:\n"
          f"- Use base features: {args.use_base_features}\n"
          f"- Use text: {args.use_text}\n"
          f"- Use audio: {args.use_audio}\n"
          f"- Use interaction: {interaction_active}\n"
          f"- Use PCA: {args.use_pca}\n"
          f"- PCA threshold: {args.pca_threshold}\n"
          f"- Number of jobs: {args.num_jobs}\n"
          f"- Optimize alpha: {args.optimize_alpha}\n"
          f"- Alpha range: 10^{args.alpha_min} to 10^{args.alpha_max} with {args.num_alphas} values\n"
          f"- Number of LOPO iterations: {args.n_lopo if args.n_lopo is not None else 'num_participants'}\n"
          f"- Correlation minimum: {args.corrmin}\n"
          f"- Number of CV splits: {args.n_splits if args.n_splits is not None else 'num_participants'}\n"
          f"- Normalize alphas: {args.normalpha}\n"
          f"- Use correlation metric: {args.use_corr}\n"
          f"- Return weights: {args.return_wt}\n"
          f"- Normalize response: {args.normalize_resp}\n"
          f"- LOPO with replacement: {args.with_replacement}\n"
          f"- Results directory: {args.results_dir if args.results_dir else 'default'}\n"
          f"- Text embedding type: {args.text_embedding_type}\n"
          f"- Included tasks: {', '.join(args.include_tasks)}")

    paths = analysis_helpers.get_paths(text_embedding_type=args.text_embedding_type)
    participant_list = os.listdir(paths["data_path"])

    icbm = datasets.fetch_icbm152_2009()
    mask_path = icbm['mask']
    # Load the mask as a Nifti image object
    mask = image.load_img(mask_path)
    
    exemple_data = nib.load("data/example_fmri/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
    resampled_mask = resample_to_img(mask, exemple_data, interpolation='nearest')

    stim_df, resp, ids_list, col_groups = analysis_helpers.load_dataset(args, paths, participant_list, resampled_mask)

    # Build feature string early (needed for valphas path)
    features_used = []
    if args.use_text:
        features_used.append("text")
    if args.use_audio:
        features_used.append("audio")
    if args.use_base_features:
        features_used.append("base")
    feature_str = "_".join(features_used) if features_used else "nofeatures"

    # Set alphas based on arguments
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.num_alphas)

    # Handle precomputed alphas
    if not args.optimize_alpha:
        valphas_path = os.path.join(args.results_dir, f"valphas_{feature_str}_{args.text_embedding_type}.npy")
        if not os.path.exists(valphas_path):
            raise ValueError(f"Must provide a valid precomputed valphas. Expected: {valphas_path}")
        valphas = np.load(valphas_path)
        ridge_logger.info(f"Using precomputed valphas at {valphas_path}")
    else:
        valphas = None

    # Set n_lopo and n_splits based on arguments or default to number of participants
    n_lopo = args.n_lopo if args.n_lopo is not None else len(participant_list)
    n_splits = args.n_splits if args.n_splits is not None else len(participant_list)

    # Perform ridge regression with LOO CV
    weights, corrs, valphas, fold_corrs, _ = ridge_cv(
        stim_df=stim_df,
        resp=resp,
        alphas=alphas,
        participant_ids=ids_list,
        col_groups=col_groups,
        use_pca=args.use_pca,
        pca_threshold=args.pca_threshold,
        n_lopo=n_lopo,
        corrmin=args.corrmin,
        n_splits=n_splits,
        singcutoff=1e-10,
        normalpha=args.normalpha,
        use_corr=args.use_corr,
        return_wt=args.return_wt,
        normalize_resp=args.normalize_resp,
        n_jobs=args.num_jobs,
        with_replacement=args.with_replacement,
        optimize_alpha=args.optimize_alpha,
        valphas=valphas,
        logger=ridge_logger
    )
    
    # Set results directory
    os.makedirs(args.results_dir, exist_ok=True)

    
    # Save corrs (mean correlations across folds) in flattened space
    corr_map_path = os.path.join(args.results_dir, f"correlation_map_flat_{feature_str}_{args.text_embedding_type}_{args.n_splits}.npy")
    np.save(corr_map_path, corrs)
    np.save(os.path.join(args.results_dir, f"folds_correlation_map_flat_{feature_str}_{args.text_embedding_type}_{args.n_splits}.npy"), fold_corrs)
    ridge_logger.info(f"Saved flattened correlations to {corr_map_path}")

    # Save valphas
    if args.optimize_alpha:
        result_file_valphas = os.path.join(args.results_dir, f"valphas_{feature_str}_{args.text_embedding_type}.npy")
        np.save(result_file_valphas, valphas)
        ridge_logger.info(f"Saved valphas to {result_file_valphas}")
    
    end_time = time.time()
    print("Total r2: %d" % sum(corrs * np.abs(corrs)))
    print(f"Analysis completed in {(end_time - start_time) / 60:.2f} minutes.")

# Run the script
if __name__ == "__main__":
    main()