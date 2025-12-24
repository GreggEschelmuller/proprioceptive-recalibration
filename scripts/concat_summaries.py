from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml


def load_cfg(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python scripts/concat_summaries.py <config.yaml>")

    cfg_path = Path(sys.argv[1])
    cfg = load_cfg(cfg_path)

    group_col = cfg.get("group_col", "config")
    inputs = cfg.get("inputs", [])
    if not inputs:
        raise ValueError("No inputs provided in YAML under 'inputs'.")

    dfs: list[pd.DataFrame] = []
    for item in inputs:
        label = item["label"]
        path = Path(item["path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")

        df = pd.read_csv(path)

        # add grouping column (and keep provenance)
        df[group_col] = label
        df.insert(0, group_col, df.pop(group_col))  # move to front

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    out_path = Path(cfg.get("out_csv", "outputs/concatenated_summary.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, na_rep="NaN")

    print(f"Saved: {out_path}")
    print("Shape:", out.shape)
    print(out[group_col].value_counts(dropna=False))


if __name__ == "__main__":
    main()