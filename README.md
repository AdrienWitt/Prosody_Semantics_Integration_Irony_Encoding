# Prosody × Semantics integration in irony & sarcasm — fMRI encoding

Voxel-wise ridge encoding of prosodic (openSMILE) and semantic (CamemBERTa v2)
features during irony/sarcasm comprehension. This repository is the minimal,
documented set of scripts needed to reproduce the analyses reported in the paper.

The models it covers are those reported in the paper: **openSMILE** (eGeMAPSv02)
audio features, and **cross-attention** (context) and **statement-only** text
embeddings.

## Scripts (pipeline order)

| Script | Role |
|---|---|
| `create_fmri_files.py` | Build one HRF-weighted fMRI `.npy` per trial from the preprocessed niftis (confound + high-pass + z-score). |
| `audio_text_embeddings.py` | Generate text (`cross_attention`, `statement_only`) and audio (`opensmile`) embeddings. |
| `dataset.py` | Assemble the design matrix (base one-hot + text/audio embeddings, optional interaction) and load the fMRI targets. |
| `ridge_cv.py` | Ridge regression with GroupKFold CV + LOPO per-voxel alpha optimization. |
| `analysis_helpers.py` | Paths and dataset-loader wrapper. |
| `encoding.py` | Main runner — fits one model, saves the CV correlation map and per-voxel alphas (`valphas`). |
| `permutation_test.py` | Block-permutation null distributions (both-blocks + Draper–Stoneman block-restricted). |
| `results_analyses.py` | FDR + top-percentile clustering → cluster tables (`.xlsx`) and thresholded maps (`.nii`). |
| `diagnostic_permutations.py` | Supplementary permutation-diagnostic figure + summary table. |

## Setup

```bash
pip install -r requirements.txt
```

Every script resolves paths relative to the repo, so a fresh clone runs without
editing any file. Place the data (available from the repository listed in the
paper's Data Availability statement) under the repo as:

```
data/preprocessed/   # preprocessed niftis, one p* folder per participant  → create_fmri_files.py
data/text/           # contexts/ and statements/                            → audio_text_embeddings.py
data/audio/          # stimulus .wav files                                  → audio_text_embeddings.py
data/behavioral/     # per-participant Resultfile_* tables
data/fmri/           # per-trial .npy, written by create_fmri_files.py
embeddings/          # written by audio_text_embeddings.py
```

`data/`, `embeddings/` and `results*/` are git-ignored. If your data lives
elsewhere, override the defaults instead of editing the scripts:

```bash
python create_fmri_files.py --fmri-dir /path/to/preprocessed --output-dir /path/to/out
python audio_text_embeddings.py --text-dir /path/to/text --audio-dir /path/to/audio
```

Run `--help` on either script for the full set of flags. All other scripts
resolve paths via `analysis_helpers.get_paths()`.

## Reproduce the pipeline

Shared settings used in the paper: `--use_pca --pca_threshold 0.55`
(must be identical in `encoding.py` and `permutation_test.py`), `--n_splits 5`,
`--include_tasks irony sarcasm`, results under `results_irosar_interaction/`.

### 1. Embeddings
```bash
python audio_text_embeddings.py     # text_cross_attention, text_statement_only, audio_opensmile
```

### 2. Encoding — one run per model (each writes its correlation map + valphas)
```bash
COMMON="--use_base_features --use_pca --pca_threshold 0.55 --n_splits 5 \
        --optimize_alpha --include_tasks irony sarcasm \
        --results_dir results_irosar_interaction"

# context-attention models
python encoding.py --use_text            $COMMON --text_embedding_type cross_attention
python encoding.py --use_audio           $COMMON
python encoding.py --use_text --use_audio --use_interaction $COMMON --text_embedding_type cross_attention

# statement-only models (no-context comparison)
python encoding.py --use_text            $COMMON --text_embedding_type statement_only
python encoding.py --use_text --use_audio --use_interaction $COMMON --text_embedding_type statement_only
```
`--results_dir results_irosar_interaction` is idempotent: combined (`--use_interaction`)
runs keep the name, unimodal runs write into the same folder so every map lives together.

### 3. Permutations — null distributions (reuse the valphas from step 2)
```bash
PERM="--use_base_features --use_pca --pca_threshold 0.55 --n_splits 5 \
      --include_tasks irony sarcasm --results_dir results_irosar_interaction --n_perms 1000"

# whole-model nulls (both blocks shuffled): text, audio, text+audio
python permutation_test.py --use_text --use_audio $PERM --text_embedding_type cross_attention \
       --include_mod text audio text_audio
# block-restricted (Draper–Stoneman) nulls for the two conditional contributions
python permutation_test.py --use_text --use_audio --use_interaction $PERM \
       --text_embedding_type cross_attention --shuffle_block audio
python permutation_test.py --use_text --use_audio --use_interaction $PERM \
       --text_embedding_type cross_attention --shuffle_block text
# repeat the three commands with --text_embedding_type statement_only
```
Operational flags for long HPC runs are available: `--perm_start/--perm_end`
(split across jobs), `--save_every` + `--resume` (checkpointing), `--merge`
(combine chunk files), `--voxel_threshold` (skip near-zero voxels).

### 4. Cluster tables & maps
```bash
python results_analyses.py          # edit the SETTINGS block at the top if needed
```
Writes to `results_irosar_interaction/tables/opensmile/perm_split5_top<pct>_k<k>_conn<c>/`
(`.xlsx` cluster tables + thresholded `.nii` per model, plus `summary.xlsx`).

### 5. Supplementary diagnostics
```bash
python diagnostic_permutations.py
```

`ridge_cv.py` is adapted from https://github.com/HuthLab/deep-fMRI-dataset.
