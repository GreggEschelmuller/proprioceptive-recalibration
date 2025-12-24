from __future__ import annotations
import sys
from pathlib import Path
import yaml
from prop_recal.pipelines.run import run  # pipeline entrypoint


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def run_single_config(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)
    label = cfg.get("message", cfg_path.name)

    print(f"\n=== Running analysis: {label} ===")
    print(f"Config: {cfg_path}")

    df = run(cfg)

    print("Loaded merged df:", df.shape)
    print(df.head())

    # optional: save merged output
    out_csv = cfg.get("outputs", {}).get(
        "out_csv",
        cfg.get("out_csv"),  # backward compatibility
    )
    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, na_rep="NaN")
        print(f"Saved: {out_path}")

    # diagnostics
    n_flagged_by_participant = (
        df["error"].isna()
        .groupby([df["participant"], df["block"]])
        .sum()
    )
    print("Flagged trials (NaN) by participant × block:")
    print(n_flagged_by_participant)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage:\n"
            "  uv run python scripts/run.py <config.yaml>\n"
            "  uv run python scripts/run.py <batch.yaml>"
        )

    path = Path(sys.argv[1])
    cfg = load_config(path)

    # ---- batch mode ----
    if "configs" in cfg:
        cfg_paths = [Path(p) for p in cfg["configs"]]

        for cfg_path in cfg_paths:
            run_single_config(cfg_path)

    # ---- single-config mode ----
    else:
        run_single_config(path)


if __name__ == "__main__":
    main()