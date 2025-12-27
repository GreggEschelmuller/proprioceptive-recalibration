from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import yaml
import sys
from pathlib import Path
from matplotlib.figure import Figure
from prop_recal.stats import bootstrap_mean_ci

# plot parameters
plt.rcParams.update({
    # Font settings
    'font.size': 12,              # Base font size
    'font.weight': 'bold',        # Bold text
    'font.family': 'sans-serif',  # Arial/Helvetica is standard for papers
    
    # Axes settings
    'axes.labelsize': 14,         # X/Y label size
    'axes.labelweight': 'bold',   # X/Y label weight
    'axes.titlesize': 16,         # Title size
    'axes.titleweight': 'bold',   # Title weight
    'axes.linewidth': 1.5,        # Thicker axes spines
    
    # Tick settings
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 1.5,     # Thicker tick marks
    'ytick.major.width': 1.5,
    
    # Legend settings
    'legend.fontsize': 12,
    'legend.frameon': False,      # Remove the box around the legend
    
    # Output resolution
    'figure.dpi': 600,            # High resolution for screen display
    'savefig.dpi': 600,           # High resolution for saved images
})


_RASTER_FORMATS = {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"}

def _normalize_formats(formats) -> list[str]:
    if formats is None:
        return ["png"]

    # If someone passes "png" or "png,svg" by mistake, recover safely.
    if isinstance(formats, str):
        s = formats.strip()
        if "," in s:
            formats = [x.strip() for x in s.split(",") if x.strip()]
        else:
            formats = [s]

    out: list[str] = []
    for fmt in formats:
        fmt = str(fmt).lower().strip().lstrip(".")
        if fmt:
            out.append(fmt)
    return out


def save_figure_multi_format(
    fig: Figure,
    base_path: Path,
    *,
    formats,
    dpi: int = 300,
    bin_number: int | None = None,
) -> None:
    base_path = Path(base_path)

    if bin_number is not None:
        base_path = base_path.with_name(f"{base_path.name}_n{bin_number}")

    base_path.parent.mkdir(parents=True, exist_ok=True)

    formats_norm = _normalize_formats(formats)

    for fmt in formats_norm:
        out_path = base_path.with_suffix(f".{fmt}")

        save_kwargs = {}
        if fmt in _RASTER_FORMATS:
            save_kwargs["dpi"] = dpi

        fig.savefig(out_path, **save_kwargs)
        print(f"Saved: {out_path}")



def plot_value_with_ci(
    summary: pd.DataFrame,
    *,
    block_col: str = "block",
    block_order: list[str],
    trials_per_block: int = 100,
    y_label: str = "Value",
    x_label: str = "Trial (concatenated across blocks)",
    title: str | None = None,
    ylim: tuple[float] | None = None, 
    out_path: Path | None = None,
    fig_formats: list[str] = ["png"],
    dpi: int = 600,
    x_col: str | None = None,
    y_col: str = "mean",
    ci_lo_col: str = "ci_lo",
    ci_hi_col: str = "ci_hi",
    alpha_band: float = 0.4,
    bin_number: int | None = None,
    figsize: tuple[float, float] = (8, 5),
):
    """
    Plot a value +/- CI bands from a summary DataFrame.

    Supports both:
      - single-trial summaries (default x = "_global_trial")
      - binned summaries (e.g., x = "_global_trial_center")

    Parameters
    ----------
    summary : pd.DataFrame
        Must contain columns: y_col, ci_lo_col, ci_hi_col, block_col, and an x column.
    x_col : str | None
        If None, uses "_global_trial" if present, else "_global_trial_center" if present.
    y_col, ci_lo_col, ci_hi_col : str
        Columns to plot for the metric and its CI.
        Examples:
          - mean error: y_col="mean", ci_lo_col="ci_lo", ci_hi_col="ci_hi"
          - within-subject VE: y_col="ve_mean", ci_lo_col="ve_ci_lo", ci_hi_col="ve_ci_hi"
    """
    # Decide x column automatically if not provided
    if x_col is None:
        if "_global_trial" in summary.columns:
            x_col = "_global_trial"
        elif "_global_trial_center" in summary.columns:
            x_col = "_global_trial_center"
        else:
            raise KeyError(
                "Could not infer x_col. Provide x_col explicitly, e.g. "
                "x_col='_global_trial' or x_col='_global_trial_center'."
            )

    required = {x_col, y_col, ci_lo_col, ci_hi_col, block_col}
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(f"summary missing columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=figsize)

    for block in block_order:
        sub = summary[summary[block_col] == block].sort_values(x_col)
        if sub.empty:
            continue

        x = sub[x_col].to_numpy()
        y = sub[y_col].to_numpy()
        lo = sub[ci_lo_col].to_numpy()
        hi = sub[ci_hi_col].to_numpy()

        ax.plot(x, y, label=str(block))
        ax.fill_between(x, lo, hi, alpha=alpha_band)

    # Block boundaries (still meaningful on concatenated axis)
    for i in range(1, len(block_order)):
        ax.axvline(i * trials_per_block + 0.5, linestyle="--", linewidth=1)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title=block_col)

    if title:
        ax.set_title(title)

    sns.despine()

    fig.tight_layout()

    if out_path is not None:
        save_figure_multi_format(
            fig,
            base_path=Path(out_path),
            formats=fig_formats,
            dpi=dpi,
            bin_number=bin_number,
        )

    return fig, ax



def plot_recalibration(
    df_subj: pd.DataFrame,
    *,
    a_col: str = "a_mean",
    b_col: str = "b_mean",
    diff_col: str = "diff",
    participant_col: str = "participant",
    block_a_label: str = "Base",
    block_b_label: str = "Post",
    title: str | None = None,
    y_label: str = "Mean Error (Degrees)",
    diff_y_label: str = "Difference Scores (Degrees)",
    ylim: tuple[float, float] | None = None,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0,
    plot_offset: float = 0.05,
    inset_rect: tuple[float, float, float, float] = (0.75, 0.40, 0.20, 0.50),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Paired Base vs Post plot with inset diff-score panel, styled like your legacy figure.

    Expects df_subj to contain:
      - a_col (baseline mean)
      - b_col (post mean)
      - diff_col (b - a)
    """
    required = {a_col, b_col, diff_col}
    missing = required - set(df_subj.columns)
    if missing:
        raise KeyError(f"df_subj missing columns: {sorted(missing)}")

    # Keep only paired rows for the main plot
    paired = df_subj[[participant_col, a_col, b_col, diff_col]].copy()
    paired = paired.dropna(subset=[a_col, b_col])

    base = paired[a_col].to_numpy(dtype=float)
    post = paired[b_col].to_numpy(dtype=float)
    diffs = paired[diff_col].to_numpy(dtype=float)

    n_valid = len(base)
    rng = np.random.default_rng(seed)

    # group means
    base_mean = float(np.mean(base)) if n_valid else np.nan
    post_mean = float(np.mean(post)) if n_valid else np.nan
    diff_mean = float(np.mean(diffs)) if n_valid else np.nan

    # bootstrap CI for diffs (this is what your legacy inset shows)
    diff_lo, diff_hi = bootstrap_mean_ci(diffs, n_boot=n_boot, ci=ci_level, rng=rng)

    # ---- main plot ----
    fig, ax = plt.subplots(figsize=(8, 6))

    x_base, x_post = 1.0, 2.0

    # subject paired lines (grey)
    for i in range(n_valid):
        ax.plot(
            [x_base - plot_offset, x_post + plot_offset],
            [base[i], post[i]],
            color="gray",
            alpha=0.3,
            zorder=1,
        )

    # subject dots (blue baseline, green post)
    ax.plot(
        [x_base - plot_offset] * n_valid,
        base,
        linestyle="None",
        marker=".",
        markersize=10,
        alpha=0.4,
        label=f"{block_a_label} Bias",
        zorder=2,
    )
    ax.plot(
        [x_post + plot_offset] * n_valid,
        post,
        linestyle="None",
        marker=".",
        markersize=10,
        alpha=0.4,
        label=f"{block_b_label} Bias",
        zorder=2,
    )

    # overall mean (black line + dots)
    ax.plot(
        [x_base, x_post],
        [base_mean, post_mean],
        color="black",
        linewidth=2,
        marker="o",
        markersize=6,
        label="Overall Mean",
        zorder=3,
    )

    # formatting like legacy
    ax.set_xlim(0.8, 3.2)
    ax.set_ylim(ylim)
    ax.set_xticks([x_base, x_post])
    ax.set_xticklabels([block_a_label, block_b_label])
    ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    # ---- inset (difference scores) ----
    inset_ax = fig.add_axes(list(inset_rect))

    # jittered x in [0.1, 0.9] like legacy
    x_jitter = rng.uniform(0.1, 0.9, size=n_valid)

    inset_ax.plot(
        x_jitter,
        diffs,
        linestyle="None",
        marker=".",
        markersize=8,
        alpha=0.6,
        color="purple",
    )

    # mean diff point + vertical CI line at x=-0.4
    inset_ax.plot(
        -0.4,
        diff_mean,
        color="black",
        marker="o",
        markersize=5,
        zorder=3,
    )
    inset_ax.plot(
        [-0.4, -0.4],
        [diff_lo, diff_hi],
        color="black",
        linewidth=2,
        zorder=3,
    )

    inset_ax.axhline(0, color="black", linestyle="--", linewidth=1)
    inset_ax.set_xlim(-1.0, 1.5)
    inset_ax.set_xticks([])
    inset_ax.set_ylabel(diff_y_label)
    sns.despine()

    return fig, ax


