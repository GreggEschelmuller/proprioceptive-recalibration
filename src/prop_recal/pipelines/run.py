from __future__ import annotations

from pathlib import Path
import pandas as pd
from prop_recal.io import load_all_participants_block_summaries  
from prop_recal.preprocess import apply_filters
from prop_recal.plotting import plot_value_with_ci
from prop_recal.stats import summarize_mean_ci_by_trial_bin



def run(cfg: dict) -> pd.DataFrame:
    data_dir = Path(cfg["data_dir"])
    participants = list(cfg["participants"])
    blocks = list(cfg["blocks"])

    df = load_all_participants_block_summaries(
        data_dir=data_dir,
        participants=participants,
        blocks=blocks,
    )

    df = apply_filters(df, filters=cfg.get("filters"))

    fig_path = Path(cfg.get("fig_path", "reports/figures/mean_error_by_trial.png"))

    summary = summarize_mean_ci_by_trial_bin(
        df,
        error_col="error",
        trial_col="trial_num",
        block_col="block",
        participant_col="participant",
        block_order=cfg["blocks"],
        trials_per_block=cfg.get("trials_per_block", 100),
        n_boot=cfg["plot"]["n_boot"],
        ci_level=cfg["plot"]["ci_level"],
        seed=cfg["plot"].get("seed", 0),
        bin_size=cfg['bin_size'],
    )

    plot_value_with_ci(
        summary,
        block_col="block",
        block_order=cfg["blocks"],
        trials_per_block=cfg.get("trials_per_block", 100),
        out_path=fig_path,
        bin_number=cfg['bin_size'],
    )


    return df
