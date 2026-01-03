from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from prop_recal.stats import (
    summarize_mean_ci_by_trial_bin,
    summarize_recalibration_two_blocks,
    summarize_within_subject_ve_ci_by_trial_bin,
    summarize_recalibration_ve_two_blocks,
)
from prop_recal.plotting import plot_value_with_ci, plot_recalibration, save_figure_multi_format


def run_trial_curve_and_recalibration_figure(df: pd.DataFrame, *, cfg: dict) -> None:
    blocks_dict: dict[str, str] = cfg["blocks"]
    blocks: list[str] = list(blocks_dict.keys())
    block_labels: dict[str, str] = blocks_dict

    plot_cfg = cfg.get("trial_plot", {})
    outputs_cfg = cfg.get("outputs", {})
    recal_cfg = cfg["recalibration"]

    # ---- trial-curve settings ----
    trials_per_block = int(cfg.get("trials_per_block", 100))
    bin_size = int(cfg.get("bin_size", 5))
    n_boot = int(plot_cfg.get("n_boot", 10_000))
    ci_level = float(plot_cfg.get("ci_level", 0.95))
    seed = int(cfg.get("seed", 0))

    xlabel = plot_cfg.get("xlabel", "Trials")
    ylim_trial = plot_cfg.get("ylim", None)

    # ---- recalibration settings ----
    block_a, block_b = recal_cfg["blocks"]
    n_first = int(recal_cfg.get("first_n", 5))
    min_valid = int(recal_cfg.get("min_valid", n_first))
    ylim_recal = recal_cfg.get("ylim", None)

    # ---- output ----
    fig_path = outputs_cfg.get("fig_path_trial_plus_recal", "reports/figures/trial_plus_recal")
    fig_path_ve = outputs_cfg.get("fig_path_trial_plus_recal_ve", "reports/figures/trial_plus_recal_ve")
    fig_formats = outputs_cfg.get("fig_formats", ["png"])
    dpi = outputs_cfg.get("dpi", 600)

    # ---- compute trial-curve summary (mean error) ----
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

    # ---- compute recalibration subject summary ----
    df_subj = summarize_recalibration_two_blocks(
        df,
        participant_col=recal_cfg.get("participant_col", "participant"),
        block_col=recal_cfg.get("block_col", "block"),
        trial_col=recal_cfg.get("trial_col", "trial_num"),
        value_col=recal_cfg.get("value_col", "error"),
        block_a=block_a,
        block_b=block_b,
        n=n_first,
        min_valid=min_valid,
    )

    # ---- make combined figure ----
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    plot_value_with_ci(
        summary_mean,
        block_col="block",
        block_order=blocks,
        block_labels=block_labels,  
        trials_per_block=trials_per_block,
        y_label=plot_cfg.get("ylabel_mean", "Constant error (degrees)"),
        x_label=xlabel,
        title=plot_cfg.get("title", None),
        ylim=ylim_trial,
        x_col="_global_trial_center",
        y_col="mean_mean",
        ci_lo_col="mean_ci_lo",
        ci_hi_col="mean_ci_hi",
        bin_number=bin_size,
        ax=axs[0],
    )

    # inset_rect is *figure* coordinates; these values usually land nicely over the right panel
    plot_recalibration(
        df_subj,
        block_a_label="Baseline",
        block_b_label="Post",
        title="Recalibration",
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        ylim=ylim_recal,
        inset_rect=(0.86, 0.3, 0.12, 0.5),
        ax=axs[1],
    )

    fig.tight_layout()

    save_figure_multi_format(
        fig,
        base_path=Path(fig_path),
        formats=fig_formats,
        dpi=dpi,
        bin_number=bin_size,
    )

# -----------------------------------------------------------------------------------------------


    # ---- compute trial-curve summary (mean error) ----
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

    # ---- compute recalibration subject summary ----
    df_subj_ve = summarize_recalibration_ve_two_blocks(
        df,
        participant_col=recal_cfg.get("participant_col", "participant"),
        block_col=recal_cfg.get("block_col", "block"),
        trial_col=recal_cfg.get("trial_col", "trial_num"),
        value_col=recal_cfg.get("value_col", "error"),
        block_a=block_a,
        block_b=block_b,
        n=n_first,
        min_valid=min_valid,
    )

    # ---- make combined figure ----
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    plot_value_with_ci(
        summary_ve,
        block_col="block",
        block_order=blocks,
        block_labels=block_labels,  
        trials_per_block=trials_per_block,
        y_label=plot_cfg.get("ylabel_ve", "Variable error (SD, degrees)"),
        x_label=xlabel,
        title=plot_cfg.get("title", None),
        ylim=ylim_trial,
        x_col="_global_trial_center",
        y_col="ve_mean",
        ci_lo_col="ve_ci_lo",
        ci_hi_col="ve_ci_hi",
        bin_number=bin_size,
        ax=axs[0],
    )

    # inset_rect is *figure* coordinates; these values usually land nicely over the right panel
    plot_recalibration(
        df_subj_ve,
        block_a_label="Baseline",
        block_b_label="Post",
        title="Recalibration",
        n_boot=n_boot,
        ci_level=ci_level,
        seed=seed,
        ylim=ylim_recal,
        inset_rect=(0.86, 0.3, 0.12, 0.5),
        ax=axs[1],
    )

    fig.tight_layout()

    save_figure_multi_format(
        fig,
        base_path=Path(fig_path_ve),
        formats=fig_formats,
        dpi=dpi,
        bin_number=bin_size,
    )