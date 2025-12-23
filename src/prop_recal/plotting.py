from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import yaml
import sys
from pathlib import Path

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
    'figure.dpi': 300,            # High resolution for screen display
    'savefig.dpi': 300,           # High resolution for saved images
})


def plot_value_with_ci(
    summary: pd.DataFrame,
    *,
    block_col: str = "block",
    block_order: list[str],
    trials_per_block: int = 100,
    y_label: str = "Value",
    x_label: str = "Trial (concatenated across blocks)",
    title: str | None = None,
    out_path: Path | None = None,
    dpi: int = 300,
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

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title=block_col)

    if title:
        ax.set_title(title)

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        if bin_number is not None:
            out_path = out_path.with_name(f"{out_path.stem}_n{bin_number}{out_path.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi)

    return fig, ax
