from __future__ import annotations

from pathlib import Path
import pandas as pd

from prop_recal.stats import summarize_recalibration_two_blocks, summarize_recalibration_ve_two_blocks
from prop_recal.plotting import plot_recalibration
from prop_recal.plotting import save_figure_multi_format


def run_recalibration_first_n(df: pd.DataFrame, *, cfg: dict) -> pd.DataFrame:
    recal_cfg = cfg["recalibration"]          # required
    outputs_cfg = cfg.get("outputs", {})

    block_a, block_b = recal_cfg["blocks"]    # required: [a, b]
    n = int(recal_cfg.get("first_n", 5))
    min_valid = int(recal_cfg.get("min_valid", n))

    df_subj = summarize_recalibration_two_blocks(
        df,
        participant_col=recal_cfg.get("participant_col", "participant"),
        block_col=recal_cfg.get("block_col", "block"),
        trial_col=recal_cfg.get("trial_col", "trial_num"),
        value_col=recal_cfg.get("value_col", "error"),
        block_a=block_a,
        block_b=block_b,
        n=n,
        min_valid=min_valid,
    )

    fig_recal, ax_recal = plot_recalibration(
        df_subj,
        block_a_label="Baseline",
        block_b_label="Post",
        title="Recalibration",
        n_boot=cfg["trial_plot"]["n_boot"],
        ci_level=cfg["trial_plot"]["ci_level"],
        seed=cfg.get("seed", 0),
        ylim=recal_cfg['ylim']
    )


    df_subj_ve = summarize_recalibration_ve_two_blocks(
        df,
        participant_col=recal_cfg.get("participant_col", "participant"),
        block_col=recal_cfg.get("block_col", "block"),
        trial_col=recal_cfg.get("trial_col", "trial_num"),
        value_col=recal_cfg.get("value_col", "error"),
        block_a=block_a,
        block_b=block_b,
        n=n,
        min_valid=min_valid,
    )

    fig_recal_ve, ax_recal_ve = plot_recalibration(
        df_subj_ve,
        block_a_label="Baseline",
        block_b_label="Post",
        title="Recalibration",
        n_boot=cfg["trial_plot"]["n_boot"],
        ci_level=cfg["trial_plot"]["ci_level"],
        seed=cfg.get("seed", 0),
        ylim=recal_cfg['ylim_ve']
    )

    # ---- save recalibration figure ----
    recal_fig_path = outputs_cfg.get(
        "fig_path_recal",
        "outputs/figures/recalibration_first5"
    )
    recal_fig_path_ve = outputs_cfg.get(
        "fig_path_recal_ve",
        "outputs/figures/recalibration_ve_first5")

    save_figure_multi_format(
        fig_recal,
        base_path=Path(recal_fig_path),
        formats=outputs_cfg.get("fig_formats", ["png"]),
        dpi=outputs_cfg.get("dpi", 600),
    )

    save_figure_multi_format(
        fig_recal_ve,
        base_path=Path(recal_fig_path_ve),
        formats=outputs_cfg.get("fig_formats", ["png"]),
        dpi=outputs_cfg.get("dpi", 600),
    )

    # Save subject-level table so you can verify immediately
    out_csv = outputs_cfg.get("out_csv_recal")
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_subj.to_csv(out_csv, index=False, na_rep="NaN")
        print(f"Saved: {out_csv}")

    return df_subj