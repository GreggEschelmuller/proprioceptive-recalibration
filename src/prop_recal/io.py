from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_single_trial(path: Path) -> pd.DataFrame:
    """Loads a single trial CSV and returns a pandas DataFrame."""
    return pd.read_csv(path)


def load_single_block_data(block_path: Path) -> pd.DataFrame:
    """Loads all trial CSVs in a block directory and concatenates them into a single DataFrame."""
    return pd.read_csv(block_path)


def load_block_summary_csv(path: Path) -> pd.DataFrame:
    """Loads a single block summary CSV (already contains all trial summaries for that block)."""
    if not path.exists():
        raise FileNotFoundError(f"Block summary CSV not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Block summary CSV is empty: {path}")

    return df


def block_summary_path(data_dir: Path, participant: int, block: str) -> Path:
    """
    Construct the expected path for a block summary CSV.
    Adjust this to match your actual filename convention.
    """
    # Example convention:
    # data_dir/p{participant}/p{participant}_{block}_summary.csv
    return data_dir / f"p{participant}" / f"p{participant}_{block}.csv"


def load_all_block_summaries(
    data_dir: Path,
    participant: int,
    blocks: list[str],
) -> pd.DataFrame:
    """
    Loads all block summary CSVs for a participant and concatenates them into one DataFrame.

    Adds 'participant' and 'block' columns for provenance.
    """
    dfs: list[pd.DataFrame] = []

    for block in blocks:
        path = block_summary_path(data_dir, participant, block)
        df = load_block_summary_csv(path)

        # Provenance column: participant (robust to schema drift)
        if "participant" in df.columns:
            df["participant"] = participant
        else:
            df.insert(0, "participant", participant)

        dfs.append(df)

    if not dfs:
        raise ValueError("No blocks provided (blocks list is empty).")

    return pd.concat(dfs, ignore_index=True)
    

def load_all_participants_block_summaries(data_dir: Path, participants: list[int], blocks: list[str],) -> pd.DataFrame:
    """
    Load block summary CSVs for multiple participants and concatenate
    into a single DataFrame.
    """
    dfs: list[pd.DataFrame] = []

    for participant in participants:
        df_p = load_all_block_summaries(data_dir=data_dir, participant=participant, blocks=blocks)
        dfs.append(df_p)

    if not dfs:
        raise ValueError("No participant data loaded.")

    return pd.concat(dfs, ignore_index=True)