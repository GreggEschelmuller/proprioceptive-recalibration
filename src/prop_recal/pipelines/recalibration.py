from __future__ import annotations

from pathlib import Path
import pandas as pd

from prop_recal.plotting import plot_recalibration  # you’ll write
from prop_recal.plotting import save_figure_multi_format  # you already have
from prop_recal.recalibration import summarize_recalibration_two_blocks  # you’ll write


def run_recalibration_first_n(df: pd.DataFrame, *, cfg: dict) -> pd.DataFrame:
    """
    Compute recalibration summary (first N trials in two blocks) and save figure(s).
    Returns participant-level table.
    """
    recal_cfg = cfg.get("recalibration", {})
    outputs_cfg = cfg.get("outputs", {})

    if not recal_cfg.get("enabled", False):
        return pd.DataFrame()

    block_a, block_b = recal_cfg["blocks"]  # require exactly two names
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

    # ---- save subject-level table (optional) ----
    out_csv = outputs_cfg.get("out_csv_recal")
    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_subj.to_csv(out_csv, index=False, na_rep="NaN")
        print(f"Saved: {out_csv}")

    # ---- plot (optional) ----
    fig_base = outputs_cfg.get("fig_path_recal")  # no suffix
    if fig_base:
        fig, _ = plot_recalibration(
            df_subj,
            block_a_label=recal_cfg.get("block_a_label", block_a),
            block_b_label=recal_cfg.get("block_b_label", block_b),
            title=recal_cfg.get("title", None),
        )

        save_figure_multi_format(
            fig,
            base_path=Path(fig_base),
            formats=outputs_cfg.get("fig_formats", ["png"]),
            dpi=int(outputs_cfg.get("dpi", 300)),
            bin_number=None,
        )

    return df_subj