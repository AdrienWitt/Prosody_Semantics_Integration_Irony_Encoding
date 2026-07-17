import argparse
import os
import glob
from pathlib import Path

import numpy as np
from nilearn.image import concat_imgs
from nilearn import signal as nilearn_signal
from nilearn import image as nilearn_image
from nilearn.maskers import NiftiMasker
import pandas as pd
from nilearn.glm.first_level import compute_regressor
from joblib import Parallel, delayed

# Defaults resolve relative to the repo, matching analysis_helpers.get_paths():
# the output default is data/fmri, which is exactly where the encoding scripts
# expect the per-trial .npy files. Override with the flags below.
BASE = Path(__file__).resolve().parent

TR = 0.65


def select_files(root_folder, files_type):
    participant_folders = glob.glob(os.path.join(root_folder, 'p*'))
    participant_files = {}
    for participant_folder in participant_folders:
        participant = participant_folder[-3:]
        run_folders = glob.glob(os.path.join(participant_folder, 'RUN*'))
        run_files = {}
        for run_folder in run_folders:
            run = run_folder[-4:]
            nii_files = glob.glob(os.path.join(run_folder, f'{files_type}*.nii'))
            run_files[run] = nii_files
        participant_files[participant] = run_files
    return participant_files


def load_dataframe(participant_path):
    file_list = [f for f in os.listdir(participant_path) if f.startswith('Resultfile_p')]
    dfs = {}
    for file_name in file_list:
        full_path = os.path.join(participant_path, file_name)
        df = pd.read_csv(full_path, sep='\t')
        key = file_name[-8:-4].upper()
        dfs[key] = df
    return dfs


def find_tissue_masks(participant_folder):
    struct_dirs = glob.glob(os.path.join(participant_folder, 'STRUCTURAL_*'))
    if not struct_dirs:
        return None, None
    struct_dir = struct_dirs[0]
    wm_files  = glob.glob(os.path.join(struct_dir, 'wc2cs*.nii'))
    csf_files = glob.glob(os.path.join(struct_dir, 'wc3cs*.nii'))
    if not wm_files or not csf_files:
        return None, None
    return wm_files[0], csf_files[0]


def extract_tissue_signals(fmri_img, wm_path, csf_path):
    wm_mask  = nilearn_image.math_img("img > 0.9", img=wm_path)
    csf_mask = nilearn_image.math_img("img > 0.9", img=csf_path)
    wm_mask_res  = nilearn_image.resample_to_img(wm_mask,  fmri_img, interpolation='nearest')
    csf_mask_res = nilearn_image.resample_to_img(csf_mask, fmri_img, interpolation='nearest')
    wm_signal  = NiftiMasker(mask_img=wm_mask_res,  standardize=False).fit_transform(fmri_img).mean(axis=1, keepdims=True)
    csf_signal = NiftiMasker(mask_img=csf_mask_res, standardize=False).fit_transform(fmri_img).mean(axis=1, keepdims=True)
    return wm_signal, csf_signal


def z_score_run(fmri):
    """Z-score each voxel's time series across the run."""
    mean_time = np.mean(fmri, axis=3, keepdims=True)
    std_time = np.std(fmri, axis=3, keepdims=True)
    std_time = np.where(std_time == 0, 1, std_time)
    return (fmri - mean_time) / std_time


def process_run(participant, run_number, run_files, dfs, output_dir_fmri, folder_fmri):
    try:
        subj_dir = os.path.join(output_dir_fmri, participant)
        os.makedirs(subj_dir, exist_ok=True)

        concatenated_img = concat_imgs(run_files)
        fmri = concatenated_img.get_fdata()

        # Build confounds
        rp_files = glob.glob(os.path.join(folder_fmri, participant, run_number, 'rp_*.txt'))
        if not rp_files:
            print(f"Warning: No motion parameter file found for {participant} {run_number}")
            return

        mp = np.loadtxt(rp_files[0])
        mp_deriv = np.vstack([mp[:1], np.diff(mp, axis=0)])
        confounds = np.hstack([mp, mp**2, mp_deriv, mp_deriv**2])

        wm_path, csf_path = find_tissue_masks(os.path.join(folder_fmri, participant))
        if wm_path and csf_path:
            wm_signal, csf_signal = extract_tissue_signals(concatenated_img, wm_path, csf_path)
            confounds = np.hstack([confounds, wm_signal, csf_signal])
            print(f"Motion + WM/CSF regression applied for {participant} {run_number}")
        else:
            print(f"Motion regression applied for {participant} {run_number} (no WM/CSF masks found)")

        # Confound regression + high-pass first
        fmri_2d = fmri.reshape(-1, fmri.shape[-1]).T  # (timepoints, voxels)
        fmri_cleaned = nilearn_signal.clean(
            fmri_2d,
            confounds=confounds,
            standardize=False,
            high_pass=0.01,
            t_r=TR,
            ensure_finite=True
        )

        x, y, z, t = fmri.shape
        fmri_cleaned_4d = fmri_cleaned.T.reshape(x, y, z, t)

        # Z-score after cleaning
        fmri_normalized = z_score_run(fmri_cleaned_4d)

        df = dfs[run_number].copy()
        df = df.rename(columns=lambda col: col.strip())
        frame_times = np.arange(0, fmri_normalized.shape[-1] * TR, TR)

        for _, row in df.iterrows():
            context = row['Context']
            statement = row['Statement']
            task = row['task']
            start_statement = row['Real_Time_Onset_Statement']
            end_statement = row['Real_Time_End_Statement']
            duration_statement = end_statement - start_statement
            end_evaluation = row['Real_Time_End_Evaluation']

            if np.isnan(end_evaluation):
                end_evaluation = row['Real_Time_Onset_Evaluation'] + 5

            exp_condition = [np.array([start_statement]),
                             np.array([duration_statement]),
                             np.array([1.0])]
            hrf_regressor, _ = compute_regressor(
                exp_condition=exp_condition,
                hrf_model='glover',
                frame_times=frame_times,
                oversampling=16
            )

            start_scan = max(0, min(round(start_statement / TR), fmri_normalized.shape[-1] - 1))
            end_scan = max(start_scan, min(round(end_evaluation / TR), fmri_normalized.shape[-1]))

            scans = fmri_normalized[..., start_scan:end_scan]
            hrf_weights = hrf_regressor[start_scan:end_scan, 0]
            hrf_weights = np.clip(hrf_weights, 0, None)

            if scans.shape[-1] == 0 or hrf_weights.sum() == 0:
                print(f"Skipping {participant}_{task}_{context[:-4]}_{statement[:-4]}: empty window or zero weights")
                continue

            hrf_weights = hrf_weights / hrf_weights.sum()
            weighted_scans = np.average(scans, axis=-1, weights=hrf_weights)

            filename = f'{participant}_{task}_{context[:-4]}_{statement[:-4]}_statement.npy'
            np.save(os.path.join(subj_dir, filename), weighted_scans)

        print(f"Completed run {run_number} for participant {participant}")

    except Exception as e:
        print(f"Error processing {participant} {run_number}: {e}")


def _parse_args():
    p = argparse.ArgumentParser(
        description="Build one HRF-weighted fMRI .npy per trial from preprocessed niftis.")
    p.add_argument("--fmri-dir", type=Path, default=BASE / "data" / "preprocessed",
                   help="preprocessed nifti root, one p* folder per participant")
    p.add_argument("--output-dir", type=Path, default=BASE / "data" / "fmri",
                   help="where the per-trial .npy files are written")
    p.add_argument("--files-type", default="swrMF",
                   help="nifti filename prefix to select")
    p.add_argument("--n-jobs", type=int, default=3, help="parallel workers")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    participant_files = select_files(args.fmri_dir, args.files_type)

    jobs = []
    for participant, runs in participant_files.items():
        dfs = load_dataframe(os.path.join(args.fmri_dir, participant))
        for run_number, run_files in runs.items():
            jobs.append((participant, run_number, run_files, dfs,
                         args.output_dir, args.fmri_dir))

    Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_run)(*job) for job in jobs
    )

    print("Processing complete.")
