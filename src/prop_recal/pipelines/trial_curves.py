from __future__ import annotations

from pathlib import Path
import pandas as pd

from prop_recal.plotting import plot_value_with_ci
from prop_recal.stats import (
    summarize_mean_ci_by_trial_bin,
    summarize_within_subject_ve_ci_by_trial_bin,
)


def run_trial_curve_plots(df: pd.DataFrame, *, cfg: dict) -> None:
    blocks = list(cfg["blocks"])
    plot_cfg = cfg.get("trial_plot", {})
    outputs_cfg = cfg.get("outputs", {})

    trials_per_block = int(cfg.get("trials_per_block", 100))
    bin_size = int(cfg.get("bin_size", 5))

    n_boot = int(plot_cfg.get("n_boot", 10_000))
    ci_level = float(plot_cfg.get("ci_level", 0.95))
    seed = int(cfg.get("seed", 0))

    fig_path_mean = Path(outputs_cfg.get("fig_path_mean", "reports/figures/mean_error_by_trial"))
    fig_path_ve = Path(outputs_cfg.get("fig_path_ve", "reports/figures/var_error_by_trial"))
    fig_formats = outputs_cfg.get("fig_formats", ["png"])

    xlabel = plot_cfg.get("xlabel", "Trials")
    title = plot_cfg.get("title", None)
    ylim = plot_cfg.get("ylim", None)

    summary_mean = summarize_mean_ci_by_trial_bin(
        df,
        error_col="error",
        trial_col="trial_num",
        block_col="block",
        participant_col="participant",
        block_order=blocks,
        trials_per_block=trials_per_block,
        bin_size=bin_size,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
    )

    summary_ve = summarize_within_subject_ve_ci_by_trial_bin(
        df,
        error_col="error",
        trial_col="trial_num",
        block_col="block",
        participant_col="participant",
        block_order=blocks,
        trials_per_block=trials_per_block,
        bin_size=bin_size,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
    )

    plot_value_with_ci(
        summary_mean,
        block_col="block",
        block_order=blocks,
        trials_per_block=trials_per_block,
        y_label=plot_cfg.get("ylabel_mean", "Constant error (degrees)"),
        x_label=xlabel,
        title=title,
        ylim=ylim,
        out_path=fig_path_mean,
        fig_formats=fig_formats,
        bin_number=bin_size,
        x_col="_global_trial_center",
        y_col="mean_mean",
        ci_lo_col="mean_ci_lo",
        ci_hi_col="mean_ci_hi",
    )

    plot_value_with_ci(
        summary_ve,
        block_col="block",
        block_order=blocks,
        trials_per_block=trials_per_block,
        y_label=plot_cfg.get("ylabel_ve", "Variable error (SD, degrees)"),
        x_label=xlabel,
        title=title,
        ylim=ylim,
        out_path=fig_path_ve,
        fig_formats=fig_formats,
        bin_number=bin_size,
        x_col="_global_trial_center",
        y_col="ve_mean",
        ci_lo_col="ve_ci_lo",
        ci_hi_col="ve_ci_hi",
    )