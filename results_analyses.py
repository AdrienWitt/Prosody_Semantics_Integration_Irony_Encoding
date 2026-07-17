import os
import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import datasets, image
from nilearn.image import resample_to_img, load_img, coord_transform
from scipy.ndimage import label, center_of_mass, generate_binary_structure
from nibabel.affines import apply_affine
from statsmodels.stats.multitest import fdrcorrection

# ================================
# SETTINGS — only change this block
# ================================
RESULTS_DIR      = "results_irosar_interaction"
MIN_CLUSTER_SIZE = 10     # manuscript tables/figures use k≥10 (perm_split5_top99_k10_conn3)
FDR_ALPHA        = 0.05
TOP_PERCENTILE   = 99     # percentile computed on POSITIVE voxels only (directional test)
N_SPLITS         = 5      # must match --n_splits used in encoding.py
CONNECTIVITY     = 3      # cluster connectivity: 1 = 6-conn (faces), 2 = 18-conn (faces+edges),
                          # 3 = 26-conn (full 3x3x3 neighbourhood: faces+edges+corners)

EMB_PRIMARY  = "cross_attention"
EMB_COMPARE  = "statement_only"

# Audio features are openSMILE (eGeMAPSv02) throughout; the output tables live under a
# fixed "opensmile" subfolder so the manuscript / figure paths stay stable.
AUDIO_SUBDIR = "opensmile"

MODE = "all"

# ================================
# Shared resources (loaded once)
# ================================
icbm       = datasets.fetch_icbm152_2009()
example    = nib.load("data/example_fmri/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
brain_mask = resample_to_img(image.load_img(icbm['mask']), example, interpolation='nearest').get_fdata() > 0
affine     = example.affine

i_coords, j_coords, k_coords = np.where(brain_mask)
n_voxels    = len(i_coords)
ijk_to_flat = {(i_coords[v], j_coords[v], k_coords[v]): v for v in range(n_voxels)}

aal      = datasets.fetch_atlas_aal('SPM12')
aal_img  = load_img(aal.maps)
aal_inv  = np.linalg.inv(aal_img.affine)
aal_data = aal_img.get_fdata()

# ================================
# Helpers
# ================================
def load_corr(feature_str, emb):
    return np.load(f"{RESULTS_DIR}/correlation_map_flat_{feature_str}_{emb}_{N_SPLITS}.npy")

def load_perm(feature_str, emb):
    return np.load(f"{RESULTS_DIR}/permutation_results/perm_scores_{feature_str}_{emb}.npy")

def load_perm_block(feature_str, emb, block):
    """Block-restricted (Draper-Stoneman) null of the combined model, where only the
    given block was shuffled (block='audio' → text aligned; block='text' → audio aligned)."""
    return np.load(f"{RESULTS_DIR}/permutation_results/perm_scores_{feature_str}_{emb}_shuffle{block}.npy")

def _vox_label_aal(vox):
    """Return AAL label for a voxel, or None."""
    try:
        idx = int(aal_data[tuple(vox)])
        if str(idx) in aal.indices:
            return aal.labels[aal.indices.index(str(idx))]
    except Exception:
        pass
    return None

def _cluster_majority_aal(cluster_mask):
    """Check all cluster voxels in AAL space, return most frequent label or None."""
    from collections import Counter
    ijs = np.argwhere(cluster_mask)  # (N, 3) in fMRI space
    counts = Counter()
    for ijk in ijs:
        mni = apply_affine(affine, ijk)
        vox = tuple(int(round(c)) for c in coord_transform(*mni, aal_inv))
        lbl = _vox_label_aal(vox)
        if lbl is not None:
            counts[lbl] += 1
    if counts:
        best, n = counts.most_common(1)[0]
        pct = 100 * n / len(ijs)
        return f"{best} ({pct:.0f}% vox)"
    return None

def _nearest_aal(mni, max_radius_mm=15, step_mm=2):
    """Sphere search around peak until a labeled AAL voxel is found."""
    center_vox = np.array([round(c) for c in coord_transform(*mni, aal_inv)])
    vox_size   = np.abs(np.diag(np.linalg.inv(aal_inv)[:3, :3]))
    for r_mm in np.arange(step_mm, max_radius_mm + step_mm, step_mm):
        r_vox = (r_mm / vox_size).astype(int)
        for di in range(-r_vox[0], r_vox[0] + 1):
            for dj in range(-r_vox[1], r_vox[1] + 1):
                for dk in range(-r_vox[2], r_vox[2] + 1):
                    dist = np.sqrt((di*vox_size[0])**2 + (dj*vox_size[1])**2 + (dk*vox_size[2])**2)
                    if dist > r_mm:
                        continue
                    v = (int(center_vox[0]+di), int(center_vox[1]+dj), int(center_vox[2]+dk))
                    lbl = _vox_label_aal(v)
                    if lbl is not None:
                        return f"{lbl} (~{dist:.0f}mm)"
    return "Unknown"

def get_aal(mni, com_mni=None, cluster_mask=None):
    """AAL lookup: peak → CoM → cluster majority vote → nearest sphere."""
    # 1) peak
    vox = tuple(int(round(c)) for c in coord_transform(*mni, aal_inv))
    lbl = _vox_label_aal(vox)
    if lbl is not None:
        return lbl
    # 2) center of mass
    if com_mni is not None:
        vox_com = tuple(int(round(c)) for c in coord_transform(*com_mni, aal_inv))
        lbl = _vox_label_aal(vox_com)
        if lbl is not None:
            return f"{lbl} (CoM)"
    # 3) majority vote on cluster voxels
    if cluster_mask is not None:
        lbl = _cluster_majority_aal(cluster_mask)
        if lbl is not None:
            return lbl
    # 4) nearest sphere search
    return _nearest_aal(mni)

def format_pval(p):
    if p < 0.001:  return "<.001"
    elif p < 0.01: return f"{p:.3f}".lstrip("0")
    else:          return f"{p:.4f}".lstrip("0")

def build_obs_and_null(mode):
    if mode == "text":
        obs       = load_corr("text_base", EMB_PRIMARY)
        perm_null = load_perm("text_base", EMB_PRIMARY)
        label_str = f"r_text ({EMB_PRIMARY})"

    elif mode == "audio":
        obs       = load_corr("audio_base", EMB_PRIMARY)
        perm_null = load_perm("audio_base", EMB_PRIMARY)
        label_str = "r_audio"

    elif mode == "text_audio":
        obs       = load_corr("text_audio_base", EMB_PRIMARY)
        perm_null = load_perm("text_audio_base", EMB_PRIMARY)
        label_str = f"r_text_audio ({EMB_PRIMARY})"

    elif mode == "delta_text_audio":
        r_comb  = load_corr("text_audio_base", EMB_PRIMARY)
        r_text  = load_corr("text_base",       EMB_PRIMARY)
        r_audio = load_corr("audio_base",      EMB_PRIMARY)
        obs     = r_comb - np.maximum(r_text, r_audio)

        p_comb  = load_perm("text_audio_base", EMB_PRIMARY)
        p_text  = load_perm("text_base",       EMB_PRIMARY)
        p_audio = load_perm("audio_base",      EMB_PRIMARY)
        perm_null = p_comb - np.maximum(p_text, p_audio)
        label_str = f"Δr text_audio - max(text,audio) [{EMB_PRIMARY}]"

    elif mode == "delta_audio_given_text":
        # Unique audio contribution r_comb - r_text; null shuffles only the audio block.
        r_comb     = load_corr("text_audio_base", EMB_PRIMARY)
        r_text     = load_corr("text_base",       EMB_PRIMARY)
        obs        = r_comb - r_text
        perm_null  = load_perm_block("text_audio_base", EMB_PRIMARY, "audio") - r_text[np.newaxis, :]
        label_str  = f"Δr audio|text = r_comb - r_text [{EMB_PRIMARY}]"

    elif mode == "delta_text_given_audio":
        # Unique text contribution r_comb - r_audio; null shuffles only the text block.
        r_comb     = load_corr("text_audio_base", EMB_PRIMARY)
        r_audio    = load_corr("audio_base",      EMB_PRIMARY)
        obs        = r_comb - r_audio
        perm_null  = load_perm_block("text_audio_base", EMB_PRIMARY, "text") - r_audio[np.newaxis, :]
        label_str  = f"Δr text|audio = r_comb - r_audio [{EMB_PRIMARY}]"

    elif mode == "text_statement_only":
        obs       = load_corr("text_base", EMB_COMPARE)
        perm_null = load_perm("text_base", EMB_COMPARE)
        label_str = f"r_text ({EMB_COMPARE})"

    elif mode == "text_audio_statement_only":
        obs       = load_corr("text_audio_base", EMB_COMPARE)
        perm_null = load_perm("text_audio_base", EMB_COMPARE)
        label_str = f"r_text_audio ({EMB_COMPARE})"

    elif mode == "delta_audio_given_text_statement_only":
        # Unique audio contribution within the statement_only embedding model.
        r_comb     = load_corr("text_audio_base", EMB_COMPARE)
        r_text     = load_corr("text_base",       EMB_COMPARE)
        obs        = r_comb - r_text
        perm_null  = load_perm_block("text_audio_base", EMB_COMPARE, "audio") - r_text[np.newaxis, :]
        label_str  = f"Δr audio|text = r_comb - r_text [{EMB_COMPARE}]"

    elif mode == "delta_text_given_audio_statement_only":
        # Unique text contribution within the statement_only embedding model.
        # Audio block is embedding-agnostic (opensmile), so r_audio uses EMB_PRIMARY.
        r_comb     = load_corr("text_audio_base", EMB_COMPARE)
        r_audio    = load_corr("audio_base",      EMB_PRIMARY)
        obs        = r_comb - r_audio
        perm_null  = load_perm_block("text_audio_base", EMB_COMPARE, "text") - r_audio[np.newaxis, :]
        label_str  = f"Δr text|audio = r_comb - r_audio [{EMB_COMPARE}]"

    elif mode == "delta_text_audio_statement_only":
        r_comb  = load_corr("text_audio_base", EMB_COMPARE)
        r_text  = load_corr("text_base",       EMB_COMPARE)
        r_audio = load_corr("audio_base",      EMB_PRIMARY)
        obs     = r_comb - np.maximum(r_text, r_audio)

        p_comb  = load_perm("text_audio_base", EMB_COMPARE)
        p_text  = load_perm("text_base",       EMB_COMPARE)
        p_audio = load_perm("audio_base",      EMB_PRIMARY)
        perm_null = p_comb - np.maximum(p_text, p_audio)
        label_str = f"Δr text_audio - max(text,audio) [{EMB_COMPARE}]"

    elif mode == "delta_emb":
        obs       = load_corr("text_base", EMB_PRIMARY) - load_corr("text_base", EMB_COMPARE)
        perm_null = load_perm("text_base", EMB_PRIMARY) - load_perm("text_base", EMB_COMPARE)
        label_str = f"Δr {EMB_PRIMARY} - {EMB_COMPARE}"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return obs, perm_null, label_str


def _perm_pvals(obs, perm_null):
    """One-tailed permutation p-value (b + 1) / (m + 1); avoids p = 0 (Phipson & Smyth, 2010)."""
    n_perm = perm_null.shape[0]
    return ((perm_null >= obs[np.newaxis, :]).sum(axis=0) + 1) / (n_perm + 1)


def compute_pvals(mode):
    """Return (obs, pvals, label_str). 'integration' is a minimum-statistic
    conjunction of the two conditional deltas: obs = min(Δaudio|text, Δtext|audio),
    p = max(p_audio|text, p_text|audio) (Nichols et al. 2005)."""
    if mode in ("integration", "integration_statement_only"):
        suffix  = "_statement_only" if mode == "integration_statement_only" else ""
        emb_lbl = EMB_COMPARE if suffix else EMB_PRIMARY
        obs_a, null_a, _ = build_obs_and_null("delta_audio_given_text" + suffix)
        obs_t, null_t, _ = build_obs_and_null("delta_text_given_audio" + suffix)
        p_a   = _perm_pvals(obs_a, null_a)
        p_t   = _perm_pvals(obs_t, null_t)
        pvals = np.maximum(p_a, p_t)
        obs   = np.minimum(obs_a, obs_t)
        return obs, pvals, f"Integration [{emb_lbl}]: min(Δaudio|text, Δtext|audio)"

    obs, perm_null, label_str = build_obs_and_null(mode)
    return obs, _perm_pvals(obs, perm_null), label_str


def run_analysis(mode):
    print(f"\n{'='*110}")
    print(f"ANALYSIS: {mode.upper()}")
    print(f"{'='*110}")

    obs, pvals, stat_label = compute_pvals(mode)
    print(f"  {stat_label}  |  obs range [{obs.min():.4f}, {obs.max():.4f}]")
    is_delta = mode.startswith("delta") or mode.startswith("integration")

    # --- FDR over voxelwise permutation p-values ---
    reject_fdr, pvals_fdr = fdrcorrection(pvals, alpha=FDR_ALPHA)
    print(f"  FDR q<{FDR_ALPHA}: {reject_fdr.sum():,} / {len(pvals):,} significant voxels")

    # --- project to 3D ---
    obs_3d    = np.zeros(brain_mask.shape)
    reject_3d = np.zeros(brain_mask.shape, dtype=bool)
    obs_3d[brain_mask]    = obs
    reject_3d[brain_mask] = reject_fdr

    # --- always restrict to positive voxels (directional hypothesis) ---
    positive_3d = obs_3d > 0
    n_pos = int(positive_3d.sum())
    print(f"  Positive voxels (obs > 0): {n_pos:,} / {brain_mask.sum():,} "
          f"({100*n_pos/brain_mask.sum():.1f}%)")

    # --- clustering mask: always FDR-significant AND positive ---
    to_cluster = reject_3d & positive_3d
    print(f"  FDR-significant ∩ positive → {to_cluster.sum():,} voxels")

    if TOP_PERCENTILE is not None:
        positive_vals = obs[obs > 0]
        threshold     = np.percentile(positive_vals, TOP_PERCENTILE)
        to_cluster    = to_cluster & (obs_3d > threshold)
        mode_label    = f"top{(100-TOP_PERCENTILE):.1f}pct_pos_FDR{int(FDR_ALPHA*100)}"
        print(f"  ∩ top {100-TOP_PERCENTILE:.1f}% (threshold = {threshold:.6f}) "
              f"→ {to_cluster.sum():,} voxels")
    else:
        mode_label = f"FDR{int(FDR_ALPHA*100)}_pos"

    output_name = f"{mode}_{mode_label}"

    cluster_structure = generate_binary_structure(3, CONNECTIVITY)
    labeled, n_clusters = label(to_cluster, structure=cluster_structure)
    print(f"  Connected clusters before size filter: {n_clusters}")

    # --- filter clusters ---
    valid_clusters = []
    for lbl in range(1, n_clusters + 1):
        mask = (labeled == lbl)
        size = int(mask.sum())
        if size < MIN_CLUSTER_SIZE:
            continue

        vals     = obs_3d[mask]
        peak_val = vals.max()
        peak_ijk = np.array(np.where((obs_3d == peak_val) & mask))[:, 0]
        i, j, k  = peak_ijk

        flat_peak = ijk_to_flat.get((i, j, k))
        if flat_peak is None:
            continue

        valid_clusters.append({
            "label":    lbl,
            "size":     size,
            "mean_r":   float(vals.mean()),
            "peak_r":   float(peak_val),
            "peak_mni": apply_affine(affine, (i, j, k)),
            "com_mni":  apply_affine(affine, center_of_mass(mask)),
            "peak_q":   float(pvals_fdr[flat_peak]),
        })

    valid_clusters = sorted(valid_clusters, key=lambda x: (x["peak_r"], x["size"]), reverse=True)
    col_stat = "Peak Δr" if is_delta else "Peak r"
    col_mean = "Mean Δr" if is_delta else "Mean r"

    table = []
    for rank, c in enumerate(valid_clusters, 1):
        region = get_aal(c["peak_mni"], c["com_mni"], labeled == c["label"])
        table.append({
            "#":          rank,
            "Size (vox)": c["size"],
            col_mean:     round(c["mean_r"], 5),
            col_stat:     round(c["peak_r"], 5),
            "Peak FDR q": format_pval(c["peak_q"]),
            "Peak MNI":   tuple(round(x, 1) for x in c["peak_mni"]),
            "CoM MNI":    tuple(round(x, 1) for x in c["com_mni"]),
            "AAL Region": region,
        })

    df = pd.DataFrame(table)
    print(f"\n  {len(valid_clusters)} clusters kept ({mode_label}, k≥{MIN_CLUSTER_SIZE})")
    print(df.to_string(index=False))

    tables_dir = os.path.join(RESULTS_DIR, "tables", AUDIO_SUBDIR, f"perm_split{N_SPLITS}_top{TOP_PERCENTILE}_k{MIN_CLUSTER_SIZE}_conn{CONNECTIVITY}")
    os.makedirs(tables_dir, exist_ok=True)

    df.to_excel(os.path.join(tables_dir, f"{output_name}.xlsx"), index=False)

    final_map = np.zeros_like(obs_3d)
    for c in valid_clusters:
        final_map[labeled == c["label"]] = obs_3d[labeled == c["label"]]
    nib.save(nib.Nifti1Image(final_map, affine), os.path.join(tables_dir, f"{output_name}.nii"))

    print(f"\n  Saved: {tables_dir}/{output_name}.{{xlsx,nii}}")
    return df


# ================================
# Run whole-brain
# ================================
ALL_MODES = ["text", "audio", "text_audio", "text_statement_only", "text_audio_statement_only",
             "delta_text_audio", "delta_text_audio_statement_only",
             "delta_audio_given_text", "delta_text_given_audio", "integration",
             "delta_audio_given_text_statement_only", "delta_text_given_audio_statement_only",
             "integration_statement_only",
             "delta_emb"]
modes_to_run = ALL_MODES if MODE == "all" else [MODE]

all_results = {}
for m in modes_to_run:
    all_results[m] = run_analysis(m)

print(f"\n{'='*110}")
print(f"Done — {len(modes_to_run)} whole-brain analyses saved to {RESULTS_DIR}/")
print(f"{'='*110}")

# ================================
# Summary table across all models
# ================================
summary_rows = []
for m in modes_to_run:
    obs, _, label_str = compute_pvals(m)
    summary_rows.append({
        "Model":    m,
        "Min":      round(float(obs.min()), 6),
        "Max":      round(float(obs.max()), 6),
        "Mean":     round(float(obs.mean()), 6),
        "Median":   round(float(np.median(obs)), 6),
        "Std":      round(float(obs.std()), 6),
        "r>0 (%)":  round(float((obs > 0).sum() / len(obs) * 100), 1),
    })

summary_df = pd.DataFrame(summary_rows)
print(f"\n{'='*110}")
print("SUMMARY — all models")
print(f"{'='*110}")
print(summary_df.to_string(index=False))

tables_dir = os.path.join(RESULTS_DIR, "tables", AUDIO_SUBDIR, f"perm_split{N_SPLITS}_top{TOP_PERCENTILE}_k{MIN_CLUSTER_SIZE}_conn{CONNECTIVITY}")
os.makedirs(tables_dir, exist_ok=True)
summary_df.to_excel(os.path.join(tables_dir, "summary.xlsx"), index=False)
print(f"\nSaved: {tables_dir}/summary.xlsx")