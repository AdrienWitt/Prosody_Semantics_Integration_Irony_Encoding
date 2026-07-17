import os
import time
import argparse
import numpy as np
import pandas as pd
import logging
import nibabel as nib
from nilearn import datasets, image
from nilearn.image import resample_to_img
from joblib import Parallel, delayed

import analysis_helpers
from ridge_cv import ridge_cv

# ----------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Permutation test for multimodal fMRI encoding models."
    )
    parser.add_argument("--use_text", action="store_true")
    parser.add_argument("--use_audio", action="store_true")
    parser.add_argument("--use_interaction", action="store_true",
                        help="Add the prosody x semantic interaction term. Must match the encoding.py run.")
    parser.add_argument("--use_base_features", action="store_true")
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--pca_threshold", type=float, default=0.6)
    parser.add_argument("--include_tasks", type=str, nargs="+", default=["sarcasm", "irony", "prosody", "semantic", "tom"])
    parser.add_argument("--n_splits", type=int, default=5, help="Must match encoding.py.")
    parser.add_argument("--n_perms", type=int, default=1000)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--num_jobs", type=int, default=7, help="Outer parallel jobs.")
    parser.add_argument("--include_mod", type=str, nargs="+", default=["text", "audio", "text_audio"],
                               choices=["text", "audio", "text_audio"],
                               help="Modalities to fit (one or more).")
    parser.add_argument("--shuffle_block", type=str, default="both",
                               choices=["both", "text", "audio"],
                               help="Block shuffled within participant: 'both' (null for r>0), "
                                    "'audio' (Draper-Stoneman null for delta_audio|text), "
                                    "'text' (null for delta_text|audio). Block modes fit the combined model only.")
    parser.add_argument("--corrmin", type=float, default=0.0)
    parser.add_argument("--normalpha", action="store_true", default=True)
    parser.add_argument("--use_corr", action="store_true", default=True)
    parser.add_argument("--normalize_stim", action="store_true")
    parser.add_argument("--normalize_resp", action="store_true", default=True)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--text_embedding_type", type=str, default="cross_attention",
                               choices=["cross_attention", "statement_only"],
                               help="Primary text embedding type.")
    parser.add_argument("--compare_embedding_type", type=str, default=None,
                               choices=["cross_attention", "statement_only"],
                               help="Second embedding, shuffled with the same index, for a delta null.")
    parser.add_argument("--voxel_threshold", type=float, default=None,
                               help="Skip voxels whose max observed |r| is below this (they get p=1).")
    parser.add_argument("--save_every", type=int, default=0,
                               help="Checkpoint every N permutations (0 = only at end).")
    parser.add_argument("--perm_start", type=int, default=0,
                               help="First permutation index (with --perm_end, to split across jobs).")
    parser.add_argument("--perm_end", type=int, default=None,
                               help="Last permutation index (exclusive; defaults to --n_perms).")
    parser.add_argument("--merge", action="store_true",
                               help="Merge saved chunks into final files and exit.")
    parser.add_argument("--resume", action="store_true",
                               help="Resume from the latest checkpoint.")
    return parser.parse_args()


# ----------------------------------------------------------------------
# Rebuild cross-modal interaction columns from their parents after shuffling
# ----------------------------------------------------------------------
def _rebuild_interactions(df: pd.DataFrame, inter_cols: list):
    """Rebuild semantic_*_x_prosody_* columns as the product of their parents,
    after shuffling, so the interaction follows the permuted design."""
    for ic in inter_cols:
        s_col, right = ic.split("_x_prosody_")
        p_col = "prosody_" + right
        df.loc[:, ic] = df[s_col].values * df[p_col].values


# ----------------------------------------------------------------------
# One permutation: permute modality blocks independently per participant
# ----------------------------------------------------------------------
def run_one_permutation(
    stim_df: pd.DataFrame,
    resp: np.ndarray,
    ids_list: np.ndarray,
    cols_text: list,
    cols_audio: list,
    cols_combined: list,
    valphas_per_mod: dict,
    args: argparse.Namespace,
    seed: int,
    col_groups: dict = None,
    cols_interaction: list = None,
    # --- optional comparison embedding ---
    stim_df2: pd.DataFrame = None,
    cols_text2: list = None,
    cols_combined2: list = None,
    # --- voxel selection ---
    voxel_mask: np.ndarray = None,
):
    rng = np.random.RandomState(seed)
    stim_perm = stim_df.copy()
    stim_perm2 = stim_df2.copy() if stim_df2 is not None else None

    # Shuffle text and/or audio within each participant, per --shuffle_block.
    shuffle_block = getattr(args, "shuffle_block", "both")
    do_text  = shuffle_block in ("both", "text")
    do_audio = shuffle_block in ("both", "audio")

    for pid in np.unique(ids_list):
        idx = np.where(ids_list == pid)[0]

        if do_text and ("text" in args.include_mod or "text_audio" in args.include_mod or stim_perm2 is not None):
            perm_idx_text = rng.permutation(idx)
            stim_perm.loc[idx, cols_text] = stim_perm.loc[perm_idx_text, cols_text].values
            # apply the SAME shuffle to the comparison embedding
            if stim_perm2 is not None:
                stim_perm2.loc[idx, cols_text2] = stim_perm2.loc[perm_idx_text, cols_text2].values

        if do_audio and ("audio" in args.include_mod or "text_audio" in args.include_mod):
            perm_idx_audio = rng.permutation(idx)
            stim_perm.loc[idx, cols_audio] = stim_perm.loc[perm_idx_audio, cols_audio].values

    # Rebuild the interaction from its (shuffled) parents.
    if cols_interaction:
        _rebuild_interactions(stim_perm, cols_interaction)
        if stim_perm2 is not None:
            _rebuild_interactions(stim_perm2, cols_interaction)

    def _cg(cols):
        """Filter col_groups to only columns present in cols."""
        if col_groups is None:
            return {}
        s = set(cols)
        return {k: [c for c in v if c in s] for k, v in col_groups.items()}

    # Subset resp to selected voxels if mask is provided
    resp_run = resp[:, voxel_mask] if voxel_mask is not None else resp
    # Subset valphas to selected voxels
    def _subset_valphas(key):
        va = valphas_per_mod[key]
        return va[voxel_mask] if voxel_mask is not None else va

    _ridge_kwargs_base = dict(
        resp=resp_run,
        alphas=None,
        participant_ids=ids_list,
        n_lopo=0,
        n_splits=args.n_splits,
        corrmin=args.corrmin,
        singcutoff=1e-10,
        normalpha=args.normalpha,
        use_corr=args.use_corr,
        return_wt=False,
        normalize_stim=args.normalize_stim,
        normalize_resp=args.normalize_resp,
        n_jobs=1,   # outer Parallel handles parallelism; nested jobs blow up memory
        with_replacement=False,
        optimize_alpha=False,
        logger=ridge_logger,
    )

    results = {}

    if "text" in args.include_mod:
        _, corr_text, _, _, _ = ridge_cv(stim_df=stim_perm[cols_text],
                                          col_groups=_cg(cols_text),
                                          valphas=_subset_valphas("text"), **_ridge_kwargs_base)
        results["text"] = corr_text

    if "audio" in args.include_mod:
        _, corr_audio, _, _, _ = ridge_cv(stim_df=stim_perm[cols_audio],
                                           col_groups=_cg(cols_audio),
                                           valphas=_subset_valphas("audio"), **_ridge_kwargs_base)
        results["audio"] = corr_audio

    if "text_audio" in args.include_mod:
        _, corr_comb, _, _, _ = ridge_cv(stim_df=stim_perm[cols_combined],
                                          col_groups=_cg(cols_combined),
                                          valphas=_subset_valphas("text_audio"), **_ridge_kwargs_base)
        results["text_audio"] = corr_comb

    # --- Comparison embedding type (same text shuffle, independent ridge fit) ---
    if stim_perm2 is not None:
        if "text" in args.include_mod:
            _, corr_text2, _, _, _ = ridge_cv(stim_df=stim_perm2[cols_text2],
                                               col_groups=_cg(cols_text2),
                                               valphas=_subset_valphas("text_compare"), **_ridge_kwargs_base)
            results["text_compare"] = corr_text2

        if "text_audio" in args.include_mod:
            _, corr_comb2, _, _, _ = ridge_cv(stim_df=stim_perm2[cols_combined2],
                                               col_groups=_cg(cols_combined2),
                                               valphas=_subset_valphas("text_audio_compare"), **_ridge_kwargs_base)
            results["text_audio_compare"] = corr_comb2

    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    start_time = time.time()

    args = parse_arguments()

    # The interaction term only exists when both modalities are present.
    interaction_active = args.use_interaction and args.use_text and args.use_audio
    if args.use_interaction and not interaction_active:
        print("WARNING: --use_interaction requires both --use_text and --use_audio; ignored.")
    # Suffix the folder with '_interaction' (mirrors encoding.py).
    if interaction_active and args.results_dir and not args.results_dir.rstrip("/").endswith("_interaction"):
        args.results_dir = args.results_dir.rstrip("/") + "_interaction"

    print(f"Running permutation test with settings:\n"
          f"- Use base features: {args.use_base_features}\n"
          f"- Use text: {args.use_text}\n"
          f"- Use audio: {args.use_audio}\n"
          f"- Use interaction: {interaction_active}\n"
          f"- Use PCA: {args.use_pca}\n"
          f"- PCA threshold: {args.pca_threshold}\n"
          f"- Number of jobs: {args.num_jobs}\n"
          f"- Number of permutations: {args.n_perms}\n"
          f"- Random seed: {args.random_seed}\n"
          f"- Number of CV splits: {args.n_splits}\n"
          f"- Correlation minimum: {args.corrmin}\n"
          f"- Normalize alphas: {args.normalpha}\n"
          f"- Use correlation metric: {args.use_corr}\n"
          f"- Normalize stimulus: {args.normalize_stim}\n"
          f"- Normalize response: {args.normalize_resp}\n"
          f"- Included modalities: {', '.join(args.include_mod)}\n"
          f"- Included tasks: {', '.join(args.include_tasks)}\n"
          f"- Text embedding type: {args.text_embedding_type}\n"
          f"- Compare embedding type: {args.compare_embedding_type or 'None'}\n"
          f"- Voxel threshold: {args.voxel_threshold if args.voxel_threshold is not None else 'None (all voxels)'}\n"
          f"- Save every: {args.save_every if args.save_every > 0 else 'only at end'}\n"
          f"- Perm range: {args.perm_start}–{args.perm_end if args.perm_end is not None else args.n_perms}\n"
          f"- Resume: {args.resume}\n"
          f"- Results directory: {args.results_dir if args.results_dir else 'default'}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger("perm_test")
    global ridge_logger
    ridge_logger = logging.getLogger("ridge_corr")
    # Silence per-fold SVD/training logs during permutations (too noisy)
    ridge_logger.setLevel(logging.WARNING)
    logger.info("=== Starting permutation test ===")

    # --- Load primary dataset ---
    paths = analysis_helpers.get_paths(text_embedding_type=args.text_embedding_type)
    participant_list = sorted(os.listdir(paths["data_path"]))

    icbm = datasets.fetch_icbm152_2009()
    mask = image.load_img(icbm["mask"])
    example_nii = nib.load("data/example_fmri/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
    resampled_mask = resample_to_img(mask, example_nii, interpolation="nearest")

    stim_df, resp, ids_list, col_groups = analysis_helpers.load_dataset(
        args, paths, participant_list, resampled_mask
    )

    # Interaction columns belong only to the combined model — keep them out of
    # cols_text (they'd otherwise be caught by the 'semantic_' prefix below).
    cols_interaction = [c for c in stim_df.columns if "_x_prosody_" in c]
    cols_text_emb   = [c for c in stim_df.columns if c.startswith("emb_text_")]
    cols_text_base  = [c for c in stim_df.columns
                       if c.startswith(("context_", "semantic_")) and "_x_prosody_" not in c]
    cols_audio_emb  = [c for c in stim_df.columns if c.startswith("emb_audio_")]
    cols_audio_base = [c for c in stim_df.columns if c.startswith("prosody_")]

    cols_text     = cols_text_emb + cols_text_base
    cols_audio    = cols_audio_emb + cols_audio_base
    cols_combined = cols_text + cols_audio + cols_interaction

    logger.info(
        f"Primary ({args.text_embedding_type}) — "
        f"text: {len(cols_text)} features | audio: {len(cols_audio)} features | "
        f"interaction: {len(cols_interaction)} features (combined model only)"
    )
    logger.info(f"CV splits: {args.n_splits} (must match encoding.py)")

    # Block-restricted shuffle only makes sense for the combined model.
    if args.shuffle_block != "both":
        if "text_audio" not in args.include_mod:
            logger.warning("Block-restricted shuffle requires the combined model; "
                           "forcing include_mod=['text_audio'].")
        args.include_mod = ["text_audio"]
        logger.info(f"Block-restricted shuffle '{args.shuffle_block}' → fitting combined model only.")

    logger.info(f"Running with modalities: {', '.join(args.include_mod)}")

    base_suffix = "_base" if args.use_base_features else ""
    # Distinguish block-restricted nulls from the default both-blocks null on disk.
    block_suffix = "" if args.shuffle_block == "both" else f"_shuffle{args.shuffle_block}"
    emb = args.text_embedding_type

    def _load_valphas(feature_str, emb_type):
        path = os.path.join(args.results_dir, f"valphas_{feature_str}{base_suffix}_{emb_type}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Valphas file not found: {path}")
        return np.load(path)

    valphas_per_mod = {}
    if "text" in args.include_mod:
        valphas_per_mod["text"] = _load_valphas("text", emb)
    if "audio" in args.include_mod:
        valphas_per_mod["audio"] = _load_valphas("audio", emb)
    if "text_audio" in args.include_mod:
        valphas_per_mod["text_audio"] = _load_valphas("text_audio", emb)

    # --- Voxel pre-selection (optional, saves memory + compute) ---
    voxel_mask = None
    n_total_voxels = resp.shape[1]
    if args.voxel_threshold is not None:
        # Load observed correlations and keep only voxels where at least one
        # modality exceeds the threshold
        obs_max = np.zeros(n_total_voxels)
        for mod in args.include_mod:
            try:
                obs_corr = np.load(os.path.join(
                    args.results_dir,
                    f"correlation_map_flat_{mod}{base_suffix}_{emb}_{args.n_splits}.npy"
                ))
                obs_max = np.maximum(obs_max, np.abs(obs_corr))
            except FileNotFoundError:
                logger.warning(f"Could not load observed correlations for {mod}, skipping pre-selection for this modality.")

        voxel_mask = obs_max > args.voxel_threshold
        n_selected = voxel_mask.sum()
        logger.info(
            f"Voxel pre-selection: {n_selected:,} / {n_total_voxels:,} voxels "
            f"(threshold |r| > {args.voxel_threshold}). "
            f"Reduction: {100 * (1 - n_selected / n_total_voxels):.1f}%"
        )
    else:
        logger.info(f"No voxel pre-selection — running all {n_total_voxels:,} voxels.")

    # --- Load comparison embedding dataset (if requested) ---
    stim_df2 = cols_text2 = cols_combined2 = None
    emb2 = args.compare_embedding_type

    if emb2 is not None:
        logger.info(f"Loading comparison embedding: {emb2}")
        args2 = argparse.Namespace(**vars(args))  # shallow copy of args
        args2.text_embedding_type = emb2
        paths2 = analysis_helpers.get_paths(text_embedding_type=emb2)
        stim_df2, _, _, _ = analysis_helpers.load_dataset(
            args2, paths2, participant_list, resampled_mask
        )
        # audio/base columns are shared; only text embeddings differ
        cols_text_emb2  = [c for c in stim_df2.columns if c.startswith("emb_text_")]
        cols_text2      = cols_text_emb2 + cols_text_base   # base cols are the same
        cols_combined2  = cols_text2 + cols_audio + cols_interaction

        if "text" in args.include_mod:
            valphas_per_mod["text_compare"] = _load_valphas("text", emb2)
        if "text_audio" in args.include_mod:
            valphas_per_mod["text_audio_compare"] = _load_valphas("text_audio", emb2)

        logger.info(
            f"Comparison ({emb2}) — text: {len(cols_text2)} features"
        )

    # --- Determine permutation range ---
    rng = np.random.RandomState(args.random_seed)
    seeds = rng.randint(0, 2**31 - 1, size=args.n_perms)

    perm_start = args.perm_start
    perm_end = args.perm_end if args.perm_end is not None else args.n_perms
    perm_end = min(perm_end, args.n_perms)
    chunk_seeds = seeds[perm_start:perm_end]
    n_chunk = len(chunk_seeds)

    results_path = args.results_dir if args.results_dir else paths["results_path"]
    perm_dir = os.path.join(results_path, "permutation_results")
    os.makedirs(perm_dir, exist_ok=True)

    # --- Merge mode: combine chunk files and exit ---
    if args.merge:
        logger.info("=== Merge mode ===")
        # Discover all chunk files per modality key
        all_mod_keys = list(args.include_mod)
        if emb2 is not None:
            all_mod_keys += [f"{m}_compare" for m in args.include_mod]

        for key in all_mod_keys:
            emb_label = emb2 if key.endswith("_compare") else emb
            mod_label = key.replace("_compare", "") if key.endswith("_compare") else key
            pattern = f"perm_scores_{mod_label}{base_suffix}_{emb_label}{block_suffix}_chunk_"
            chunk_files = sorted([
                f for f in os.listdir(perm_dir)
                if f.startswith(pattern) and f.endswith(".npy")
            ])
            if not chunk_files:
                logger.warning(f"No chunk files found for {key} (pattern: {pattern}*)")
                continue
            arrays = [np.load(os.path.join(perm_dir, f)) for f in chunk_files]
            merged = np.vstack(arrays)
            out_file = os.path.join(perm_dir, f"perm_scores_{mod_label}{base_suffix}_{emb_label}{block_suffix}.npy")
            np.save(out_file, merged)
            logger.info(f"Merged {len(chunk_files)} chunks → {out_file} — shape {merged.shape}")
        logger.info("Merge complete.")
        return

    is_chunk = (perm_start > 0 or perm_end < args.n_perms)
    chunk_suffix = f"_chunk_{perm_start}_{perm_end}" if is_chunk else ""

    # --- Resume: reload checkpoint and skip already-done permutations ---
    n_done = 0
    resumed_scores = {}  # key → np.ndarray (n_done, n_voxels_selected)
    if args.resume:
        ref_mod = args.include_mod[0]

        # Check if final file already exists (chunk completed in a previous run)
        final_path = os.path.join(perm_dir, f"perm_scores_{ref_mod}{base_suffix}_{emb}{block_suffix}{chunk_suffix}.npy")
        if os.path.exists(final_path) and not final_path.endswith("_checkpoint.npy"):
            ref_arr = np.load(final_path)
            expected = perm_end - perm_start
            if ref_arr.shape[0] >= expected:
                logger.info(f"Final file already exists with {ref_arr.shape[0]} perms — nothing to do")
                return

        # Look for checkpoint files
        ckpt_path = os.path.join(perm_dir, f"perm_scores_{ref_mod}{base_suffix}_{emb}{block_suffix}{chunk_suffix}_checkpoint.npy")
        if os.path.exists(ckpt_path):
            ref_arr = np.load(ckpt_path)
            n_done = ref_arr.shape[0]
            logger.info(f"Resuming: found checkpoint with {n_done} permutations already done")

            # Load all modality checkpoints
            all_mod_keys = list(args.include_mod)
            if emb2 is not None:
                all_mod_keys += [f"{m}_compare" for m in args.include_mod]
            for key in all_mod_keys:
                emb_label = emb2 if key.endswith("_compare") else emb
                mod_label = key.replace("_compare", "") if key.endswith("_compare") else key
                cp = os.path.join(perm_dir, f"perm_scores_{mod_label}{base_suffix}_{emb_label}{block_suffix}{chunk_suffix}_checkpoint.npy")
                if os.path.exists(cp):
                    resumed_scores[key] = np.load(cp)
                    logger.info(f"  Loaded {cp} — shape {resumed_scores[key].shape}")
        else:
            logger.info("Resume requested but no checkpoint found — starting from scratch")

    chunk_seeds = chunk_seeds[n_done:]
    n_chunk = len(chunk_seeds)

    if n_chunk == 0:
        logger.info("All permutations already completed — nothing to do")
        return

    logger.info(f"Running permutations {perm_start + n_done}–{perm_end} ({n_chunk} remaining){' [chunk mode]' if is_chunk else ''}")

    def _run_batch(batch_seeds):
        """Run a batch of permutations."""
        return Parallel(n_jobs=args.num_jobs)(
            delayed(run_one_permutation)(
                stim_df, resp, ids_list,
                cols_text, cols_audio, cols_combined,
                valphas_per_mod, args, seed,
                col_groups=col_groups,
                cols_interaction=cols_interaction,
                stim_df2=stim_df2,
                cols_text2=cols_text2,
                cols_combined2=cols_combined2,
                voxel_mask=voxel_mask,
            )
            for seed in batch_seeds
        )

    if args.save_every > 0:
        # Run in batches, saving intermediate results
        all_perm_results = []
        for batch_start in range(0, n_chunk, args.save_every):
            batch_end = min(batch_start + args.save_every, n_chunk)
            batch_seeds_slice = chunk_seeds[batch_start:batch_end]
            abs_start = perm_start + n_done + batch_start
            abs_end = perm_start + n_done + batch_end
            logger.info(f"Running permutations {abs_start+1}–{abs_end} / {args.n_perms}...")

            batch_results = _run_batch(batch_seeds_slice)
            all_perm_results.extend(batch_results)

            # Save intermediate checkpoint (resumed + new)
            all_keys = list(all_perm_results[0].keys())
            perm_scores_so_far = {}
            for key in all_keys:
                new_scores = np.vstack([r[key] for r in all_perm_results])
                if key in resumed_scores:
                    perm_scores_so_far[key] = np.vstack([resumed_scores[key], new_scores])
                else:
                    perm_scores_so_far[key] = new_scores

            for key in all_keys:
                emb_label = emb2 if key.endswith("_compare") else emb
                mod_label = key.replace("_compare", "") if key.endswith("_compare") else key
                out_file = os.path.join(perm_dir, f"perm_scores_{mod_label}{base_suffix}_{emb_label}{block_suffix}{chunk_suffix}_checkpoint.npy")
                np.save(out_file, perm_scores_so_far[key])

            elapsed = (time.time() - start_time) / 60
            total_done = n_done + batch_end
            logger.info(f"Checkpoint saved: {total_done} perms total ({elapsed:.1f} min elapsed)")

        perm_results = all_perm_results
    else:
        perm_results = _run_batch(chunk_seeds)

    # Collect all keys and combine with resumed data
    all_keys = list(perm_results[0].keys())
    perm_scores = {}
    for key in all_keys:
        new_scores = np.vstack([r[key] for r in perm_results])
        if key in resumed_scores:
            scores_selected = np.vstack([resumed_scores[key], new_scores])
        else:
            scores_selected = new_scores
        if voxel_mask is not None:
            # Expand back to full voxel space, filling unselected voxels with 0
            n_total_done = scores_selected.shape[0]
            scores_full = np.zeros((n_total_done, n_total_voxels))
            scores_full[:, voxel_mask] = scores_selected
            perm_scores[key] = scores_full
        else:
            perm_scores[key] = scores_selected

    total_perms = n_done + n_chunk
    logger.info(f"Completed {n_chunk} new + {n_done} resumed = {total_perms} total permutations.")

    # --- Save results ---
    # Primary modalities
    for mod in args.include_mod:
        if mod in perm_scores:
            out_file = os.path.join(perm_dir, f"perm_scores_{mod}{base_suffix}_{emb}{block_suffix}{chunk_suffix}.npy")
            np.save(out_file, perm_scores[mod])
            logger.info(f"Saved {out_file} — shape {perm_scores[mod].shape}")

    # Comparison embedding modalities
    if emb2 is not None:
        for mod in args.include_mod:
            key = f"{mod}_compare"
            if key in perm_scores:
                out_file = os.path.join(perm_dir, f"perm_scores_{mod}{base_suffix}_{emb2}{block_suffix}{chunk_suffix}.npy")
                np.save(out_file, perm_scores[key])
                logger.info(f"Saved {out_file} — shape {perm_scores[key].shape}")

    # Clean up checkpoint files
    if args.save_every > 0:
        for key in all_keys:
            emb_label = emb2 if key.endswith("_compare") else emb
            mod_label = key.replace("_compare", "") if key.endswith("_compare") else key
            ckpt = os.path.join(perm_dir, f"perm_scores_{mod_label}{base_suffix}_{emb_label}{block_suffix}{chunk_suffix}_checkpoint.npy")
            if os.path.exists(ckpt):
                os.remove(ckpt)
                logger.info(f"Removed checkpoint: {ckpt}")

    logger.info(f"Total time: {(time.time() - start_time) / 60:.1f} min")


if __name__ == "__main__":
    main()
