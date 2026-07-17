"""
Permutation-test diagnostic figure for reviewers (supplementary material).

Restricted to the permutation tests behind the results we present: the unimodal
and combined encoding models (context-attention and statement-only) and the four
conditional contributions (Draper–Stoneman block-shuffled) whose conjunctions define
the two integration maps. One row of 6 panels per test:
  A  P-value histogram (null calibration)
  B  Observed vs null separation
  C  QQ-plot of -log10 p-values
  D  Example null distributions (3 representative voxels)
  E  Effect-size distribution (significant vs non-significant)
  F  Null std per voxel vs observed r (homogeneity check)

Also produces a summary CSV table.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import pandas as pd
from nilearn import datasets, image
from nilearn.image import resample_to_img
from statsmodels.stats.multitest import fdrcorrection

# ================================
# Settings
# ================================
RESULTS_DIR = "results_irosar_interaction"
EMB_PRIMARY = "cross_attention"
EMB_COMPARE = "statement_only"
N_SPLITS    = 5
FDR_ALPHA   = 0.05

# ================================
# Load brain mask
# ================================
example    = nib.load("data/example_fmri/p01_irony_CNf1_2_SNnegh4_2_statement_masked.nii.gz")
icbm       = datasets.fetch_icbm152_2009()
brain_mask = resample_to_img(
    image.load_img(icbm['mask']), example, interpolation='nearest'
).get_fdata() > 0

def load_corr(feature_str, emb):
    return np.load(f"{RESULTS_DIR}/correlation_map_flat_{feature_str}_{emb}_{N_SPLITS}.npy")

def load_perm(feature_str, emb):
    return np.load(f"{RESULTS_DIR}/permutation_results/perm_scores_{feature_str}_{emb}.npy")

def load_perm_block(feature_str, emb, block):
    """Draper–Stoneman null of the combined model where only `block` was shuffled
    (block='audio' -> text kept aligned; block='text' -> audio kept aligned)."""
    return np.load(f"{RESULTS_DIR}/permutation_results/perm_scores_{feature_str}_{emb}_shuffle{block}.npy")


# ================================
# Build observed + null for each model
# ================================
MODELS = {
    "Text (context)": {
        "obs_fn":  lambda: load_corr("text_base", EMB_PRIMARY),
        "null_fn": lambda: load_perm("text_base", EMB_PRIMARY),
    },
    "Audio": {
        "obs_fn":  lambda: load_corr("audio_base", EMB_PRIMARY),
        "null_fn": lambda: load_perm("audio_base", EMB_PRIMARY),
    },
    "Text+Audio (context)": {
        "obs_fn":  lambda: load_corr("text_audio_base", EMB_PRIMARY),
        "null_fn": lambda: load_perm("text_audio_base", EMB_PRIMARY),
    },
    "Text (statement-only)": {
        "obs_fn":  lambda: load_corr("text_base", EMB_COMPARE),
        "null_fn": lambda: load_perm("text_base", EMB_COMPARE),
    },
    "Text+Audio (statement-only)": {
        "obs_fn":  lambda: load_corr("text_audio_base", EMB_COMPARE),
        "null_fn": lambda: load_perm("text_audio_base", EMB_COMPARE),
    },
    r"$\Delta r_{\mathrm{audio\,|\,text}}$ (context)": {
        "obs_fn":  lambda: (load_corr("text_audio_base", EMB_PRIMARY)
                            - load_corr("text_base", EMB_PRIMARY)),
        "null_fn": lambda: (load_perm_block("text_audio_base", EMB_PRIMARY, "audio")
                            - load_corr("text_base", EMB_PRIMARY)[np.newaxis, :]),
    },
    r"$\Delta r_{\mathrm{text\,|\,audio}}$ (context)": {
        "obs_fn":  lambda: (load_corr("text_audio_base", EMB_PRIMARY)
                            - load_corr("audio_base", EMB_PRIMARY)),
        "null_fn": lambda: (load_perm_block("text_audio_base", EMB_PRIMARY, "text")
                            - load_corr("audio_base", EMB_PRIMARY)[np.newaxis, :]),
    },
    r"$\Delta r_{\mathrm{audio\,|\,text}}$ (stmt-only)": {
        "obs_fn":  lambda: (load_corr("text_audio_base", EMB_COMPARE)
                            - load_corr("text_base", EMB_COMPARE)),
        "null_fn": lambda: (load_perm_block("text_audio_base", EMB_COMPARE, "audio")
                            - load_corr("text_base", EMB_COMPARE)[np.newaxis, :]),
    },
    r"$\Delta r_{\mathrm{text\,|\,audio}}$ (stmt-only)": {
        "obs_fn":  lambda: (load_corr("text_audio_base", EMB_COMPARE)
                            - load_corr("audio_base", EMB_PRIMARY)),
        "null_fn": lambda: (load_perm_block("text_audio_base", EMB_COMPARE, "text")
                            - load_corr("audio_base", EMB_PRIMARY)[np.newaxis, :]),
    },
}

# Load and pre-compute stats
data = {}
for name, fns in MODELS.items():
    obs       = fns["obs_fn"]()
    perm_null = fns["null_fn"]()
    n_perms   = perm_null.shape[0]
    pvals     = ((perm_null >= obs[np.newaxis, :]).sum(axis=0) + 1) / (n_perms + 1)  # (b+1)/(m+1), avoids p=0 (Phipson & Smyth, 2010)
    # Voxels never permuted (below --voxel_threshold) have an all-zero null,
    # which would give a spuriously small p whenever obs>0. They were not tested, so mark p=1.
    never_tested = perm_null.std(axis=0) == 0
    pvals[never_tested] = 1.0
    reject, qvals = fdrcorrection(pvals, alpha=FDR_ALPHA)
    data[name] = dict(obs=obs, null=perm_null, pvals=pvals,
                       reject=reject, qvals=qvals, n_perms=n_perms)

all_names = list(data.keys())
n_models  = len(all_names)
n_cols    = 6

# ================================
# Summary table
# ================================
rows = []
for name in all_names:
    d   = data[name]
    obs = d["obs"]
    n_v = len(obs)
    rows.append({
        "Model":              name,
        "n_perm":             d["n_perms"],
        "obs_mean":           f"{obs.mean():.5f}",
        "obs_max":            f"{obs.max():.5f}",
        "null_mean":          f"{d['null'].mean():.5f}",
        "null_std (vox avg)": f"{d['null'].std(axis=0).mean():.5f}",
        "null_max":           f"{d['null'].max():.5f}",
        "p<.05 (uncorr)":    f"{(d['pvals'] < 0.05).sum():,}",
        "FDR q<.05":         f"{d['reject'].sum():,}",
        "FDR q<.05 & r>0":   f"{(d['reject'] & (obs > 0)).sum():,}",
        "% signif":           f"{100 * d['reject'].sum() / n_v:.1f}",
    })

summary_df = pd.DataFrame(rows)
print("\n" + summary_df.to_string(index=False))
summary_df.to_excel(f"{RESULTS_DIR}/permutation_diagnostic_summary.xlsx", index=False)

# ================================
# Figure: one row per model, 6 columns
# ================================
fig, axes = plt.subplots(n_models, n_cols, figsize=(n_cols * 4.2, n_models * 3.2))
if n_models == 1:
    axes = axes[np.newaxis, :]

rng = np.random.RandomState(0)

for row, name in enumerate(all_names):
    d       = data[name]
    obs     = d["obs"]
    null    = d["null"]
    pvals   = d["pvals"]
    reject  = d["reject"]
    qvals   = d["qvals"]
    n_perms = d["n_perms"]
    n_vox   = len(obs)
    is_delta = ("Delta" in name) or name.startswith("Δ")
    r_label  = "Δr" if is_delta else "r"

    # ---- A: P-value histogram ----
    ax = axes[row, 0]
    ax.hist(pvals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(0.05, color="red", ls="--", lw=1.2, label="p = .05")
    ax.axhline(n_vox / 50, color="grey", ls=":", lw=1, label="Uniform")
    ax.set_xlabel("p-value")
    ax.set_ylabel("# voxels")
    if row == 0:
        ax.set_title("A — P-value histogram", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, loc="upper right")
    ax.text(0.97, 0.80,
            f"p=min: {(pvals <= 1.0 / (n_perms + 1) + 1e-12).sum():,}\nFDR<.05: {reject.sum():,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # ---- B: Observed vs null ----
    ax = axes[row, 1]
    null_flat = null[rng.choice(n_perms, min(200, n_perms), replace=False), :].ravel()
    null_flat = rng.choice(null_flat, min(500_000, len(null_flat)), replace=False)
    ax.hist(null_flat, bins=120, density=True, color="grey", alpha=0.5, label="Null")
    ax.hist(obs, bins=120, density=True, color="steelblue", alpha=0.6, label="Observed")
    ax.axvline(0, color="k", ls="--", lw=0.6)
    ax.set_xlabel(r_label)
    ax.set_ylabel("Density")
    if row == 0:
        ax.set_title("B — Obs vs null", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6)

    # ---- C: QQ-plot ----
    ax = axes[row, 2]
    pvals_sorted = np.sort(pvals)
    floor = 0.5 / n_perms
    pvals_plot = np.maximum(pvals_sorted, floor)
    n = len(pvals_plot)
    expected = -np.log10((np.arange(1, n + 1)) / (n + 1))
    observed = -np.log10(pvals_plot)
    keep = np.sort(np.unique(np.concatenate([
        np.arange(0, min(500, n)),
        np.arange(max(0, n - 500), n),
        rng.choice(n, min(5000, n), replace=False),
    ])))
    ax.scatter(expected[keep], observed[keep], s=1, alpha=0.4, c="steelblue")
    lim = max(expected.max(), observed.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8)
    ax.set_xlabel("Expected −log₁₀(p)")
    ax.set_ylabel("Observed −log₁₀(p)")
    if row == 0:
        ax.set_title("C — QQ-plot", fontsize=10, fontweight="bold")

    # ---- D: Example null distributions (3 voxels) ----
    ax = axes[row, 3]
    idx_best = np.argmax(obs)
    sig_voxels = np.where(reject)[0]
    if len(sig_voxels) > 2:
        idx_median = sig_voxels[np.argsort(obs[sig_voxels])[len(sig_voxels) // 2]]
    else:
        idx_median = np.argsort(obs)[n_vox // 2]
    nonsig_pos = np.where(~reject & (obs > 0))[0]
    if len(nonsig_pos) > 0:
        idx_nonsig = nonsig_pos[np.argsort(obs[nonsig_pos])[-1]]
    else:
        idx_nonsig = np.argsort(obs)[n_vox // 2]

    colors_d = ["#d62728", "#ff7f0e", "#2ca02c"]
    vox_info = [
        (idx_best,   "Best"),
        (idx_median, "Med. sig."),
        (idx_nonsig, "Best non-sig."),
    ]
    for (idx, vlbl), c in zip(vox_info, colors_d):
        ax.hist(null[:, idx], bins=25, alpha=0.3, color=c, density=True)
        ax.axvline(obs[idx], color=c, lw=1.8,
                   label=f"{vlbl} ({r_label}={obs[idx]:.3f}, p={pvals[idx]:.3f})")
    ax.set_xlabel(r_label)
    ax.set_ylabel("Density")
    if row == 0:
        ax.set_title("D — Null examples", fontsize=10, fontweight="bold")
    ax.legend(fontsize=5.5, loc="upper left")

    # ---- E: Effect-size distribution (sig vs non-sig) ----
    ax = axes[row, 4]
    obs_sig    = obs[reject & (obs > 0)]
    obs_nonsig = obs[~reject & (obs > 0)]
    if len(obs_nonsig) > 0:
        ax.hist(obs_nonsig, bins=80, density=True, color="grey", alpha=0.5,
                label=f"Non-sig (n={len(obs_nonsig):,})")
    if len(obs_sig) > 0:
        ax.hist(obs_sig, bins=80, density=True, color="steelblue", alpha=0.6,
                label=f"FDR-sig (n={len(obs_sig):,})")
        ax.axvline(obs_sig.min(), color="red", ls="--", lw=1,
                   label=f"Min sig {r_label}={obs_sig.min():.4f}")
    ax.set_xlabel(f"Observed {r_label}")
    ax.set_ylabel("Density")
    if row == 0:
        ax.set_title("E — Effect sizes", fontsize=10, fontweight="bold")
    ax.legend(fontsize=5.5)

    # ---- F: Null std vs observed r ----
    ax = axes[row, 5]
    null_std_v = null.std(axis=0)
    sub = rng.choice(n_vox, min(6000, n_vox), replace=False)
    ax.scatter(obs[sub], null_std_v[sub], s=1, alpha=0.3, c="mediumpurple")
    ax.set_xlabel(f"Observed {r_label}")
    ax.set_ylabel("Null std (per voxel)")
    if row == 0:
        ax.set_title("F — Null homogeneity", fontsize=10, fontweight="bold")

    # Row label on the left
    axes[row, 0].annotate(
        name, xy=(-0.45, 0.5), xycoords="axes fraction",
        fontsize=11, fontweight="bold", ha="right", va="center", rotation=90)

fig.savefig(f"{RESULTS_DIR}/diagnostic_permutations.png", dpi=180, bbox_inches="tight")
plt.show()
print(f"\nSaved figure: {RESULTS_DIR}/diagnostic_permutations.png")
print(f"Saved table:  {RESULTS_DIR}/permutation_diagnostic_summary.xlsx")
