from pathlib import Path

from prop_recal.pipelines.run import run


def test_pipeline_runs(tmp_path: Path) -> None:
    csv_path = tmp_path / "numbers.csv"
    csv_path.write_text("value\n1\n2\n3\n")

    cfg = {
        "run_name": "test",
        "input_csv": str(csv_path),
        "output_dir": str(tmp_path / "out"),
    }

    out = run(cfg)
    assert out.exists()
    assert out.suffix == ".png"