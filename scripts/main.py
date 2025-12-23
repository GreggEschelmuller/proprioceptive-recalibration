from __future__ import annotations
import sys
from pathlib import Path
import yaml
from prop_recal.pipelines.run import run  # pipeline entrypoint


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: uv run python scripts/run.py <config.yaml>")

    config_path = Path(sys.argv[1])
    cfg = load_config(config_path)

    df = run(cfg)

    # quick “does it work?” checks
    print("Loaded merged df:", df.shape)
    print(df.head())

    # optional: save merged output so you can inspect it
    out_path = Path(cfg.get("out_csv", "outputs/merged_block_summaries.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, na_rep="NaN")
    print(f"Saved: {out_path}")

    n_flagged_by_participant = (
        df["error"].isna()
        .groupby([df["participant"], df["block"]])
        .sum()
    )
    # n_flagged_by_participant -= cfg.get("vib_trials_nf", 0)*2  # optional: exclude known invalid trials
    print(n_flagged_by_participant)

if __name__ == "__main__":
    main()