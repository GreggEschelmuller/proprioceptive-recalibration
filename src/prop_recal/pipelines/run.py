from __future__ import annotations

from pathlib import Path
import pandas as pd
from prop_recal.io import load_all_participants_block_summaries  
from prop_recal.preprocess import apply_filters
from prop_recal.plotting import plot_value_with_ci
from prop_recal.stats import summarize_mean_ci_by_trial_bin, summarize_within_subject_ve_ci_by_trial_bin


def run(cfg: dict) -> pd.DataFrame:
    data_dir = Path(cfg["data_dir"])
    participants = list(cfg["participants"])
    blocks = list(cfg["blocks"])

    plot_cfg = cfg.get("trial_plot", {})
    filters_cfg = cfg.get("filters", {})

    trials_per_block = int(cfg.get("trials_per_block", 100))
    bin_size = int(cfg.get("bin_size", 5))

    n_boot = int(plot_cfg.get("n_boot", 10_000))
    ci_level = float(plot_cfg.get("ci_level", 0.95))
    seed = int(plot_cfg.get("seed", 0))

    df = load_all_participants_block_summaries(
        data_dir=data_dir,
        participants=participants,
        blocks=blocks,
    )

    df = apply_filters(df, filters=filters_cfg)

    fig_path_mean = Path(cfg.get("fig_path_mean", "reports/figures/mean_error_by_trial.png"))
    fig_path_ve = Path(cfg.get("fig_path_ve", "reports/figures/var_error_by_trial.png"))

    summary_mean = summarize_mean_ci_by_trial_bin(
        df,
        error_col="error",
        trial_col="trial_num",
        block_col="block",
        participant_col="participant",
        block_order=blocks,
        trials_per_block=trials_per_block,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        bin_size=bin_size,
    )

    summary_ve = summarize_within_subject_ve_ci_by_trial_bin(
        df,
        error_col="error",
        trial_col="trial_num",
        block_col="block",
        participant_col="participant",
        block_order=blocks,
        trials_per_block=trials_per_block,
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        bin_size=bin_size,
    )

    # labels/titles (with defaults)
    xlabel = plot_cfg.get("xlabel", "Trials")
    title = plot_cfg.get("title", None)
    ylim = plot_cfg.get("ylim", None)  # should be [low, high] or null in YAML

    plot_value_with_ci(
        summary_mean,
        block_col="block",
        block_order=blocks,
        trials_per_block=trials_per_block,
        y_label=plot_cfg.get("ylabel_mean", plot_cfg.get("ylabel", "Error (deg)")),
        x_label=xlabel,
        title=title,
        ylim=ylim,
        out_path=fig_path_mean,
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
        y_label=plot_cfg.get("ylabel_ve", "Variable error (SD, deg)"),
        x_label=xlabel,
        title=title,
        ylim=ylim,
        out_path=fig_path_ve,
        bin_number=bin_size,
        x_col="_global_trial_center",
        y_col="ve_mean",
        ci_lo_col="ve_ci_lo",
        ci_hi_col="ve_ci_hi",
    )

    return df
