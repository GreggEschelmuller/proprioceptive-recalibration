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

    inputs = cfg.get("inputs", [])
    if not inputs:
        raise ValueError("No inputs provided in YAML under 'inputs'.")

    perturbation_col = cfg.get("perturbation_col", "perturbation")
    movement_col = cfg.get("movement_col", "movement")

    dfs: list[pd.DataFrame] = []

    for item in inputs:
        perturbation = item["perturbation"]
        movement = item["movement"]
        path = Path(item["path"])

        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")

        df = pd.read_csv(path)

        # add factor columns
        df[perturbation_col] = perturbation
        df[movement_col] = movement

        # move to front for convenience
        df.insert(0, movement_col, df.pop(movement_col))
        df.insert(0, perturbation_col, df.pop(perturbation_col))

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)

    out_path = Path(cfg.get("out_csv", "outputs/concatenated_summary.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, na_rep="NaN")

    print(f"Saved: {out_path}")
    print("Shape:", out.shape)
    print(out.groupby([perturbation_col, movement_col]).size())


if __name__ == "__main__":
    main()