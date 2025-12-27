from __future__ import annotations

from pathlib import Path
import pandas as pd

from prop_recal.io import load_all_participants_block_summaries
from prop_recal.preprocess import apply_filters

from prop_recal.pipelines.trial_curves import run_trial_curve_plots
from prop_recal.pipelines.recalibration import run_recalibration_first_n
from prop_recal.pipelines.combined_fig import run_trial_curve_and_recalibration_figure


def run(cfg: dict) -> pd.DataFrame:
    # ---- required top-level config ----
    data_dir = Path(cfg["data_dir"])
    participants = list(cfg["participants"])
    blocks = list(cfg["blocks"])

    filters_cfg = cfg.get("filters", {})
    outputs_cfg = cfg.get("outputs", {})

    out_csv = outputs_cfg.get("out_csv")
    out_csv = Path(out_csv) if out_csv else None

    # ---- load + preprocess ----
    df = load_all_participants_block_summaries(
        data_dir=data_dir,
        participants=participants,
        blocks=blocks,
    )
    df = apply_filters(df, filters=filters_cfg)

    # ---- analyses / figures (both always run) ----
    run_trial_curve_plots(df, cfg=cfg)
    run_recalibration_first_n(df, cfg=cfg)
    # new (hook)
    if cfg.get("combined_figure", {}).get("enabled", False):
        run_trial_curve_and_recalibration_figure(df, cfg=cfg)

    # ---- optional: save merged data ----
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, na_rep="NaN")
        print(f"Saved: {out_csv}")

    return df
