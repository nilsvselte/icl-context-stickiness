import csv
import json
from pathlib import Path

from when_context_sticks.paper import (
    write_arch_sweep_summary,
    write_hyperparameter_table,
)


def test_write_arch_sweep_summary_uses_average_loss_and_model_config(tmp_path: Path):
    run_dir = tmp_path / "runs" / "small" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        """
model:
  n_embd: 128
  n_layer: 8
  n_head: 8
""".strip()
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "label": "small",
                        "run_dir": str(run_dir),
                        "summary": {
                            "average_loss": 0.25,
                            "average_excess_loss": 1.5,
                            "overall_loss": 0.2,
                            "step": 99999,
                            "device": "cpu",
                            "seed": 0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_csv = tmp_path / "table_1_architecture_sweep.csv"
    write_arch_sweep_summary(manifest_path, output_csv)

    with output_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "small"
    assert row["n_embd"] == "128"
    assert row["n_layer"] == "8"
    assert row["n_head"] == "8"
    assert row["average_loss"] == "0.25"
    assert row["average_excess_loss"] == "1.5"
    assert row["final_loss"] == "0.2"


def test_write_hyperparameter_table_contains_setup_and_training_rows(tmp_path: Path):
    output_csv = tmp_path / "table_2_hyperparameters.csv"
    write_hyperparameter_table(output_csv)

    with output_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert any(
        row["parameter"] == "base_experimental_setup"
        and "What Can Transformers Learn In-Context?" in row["value"]
        for row in rows
    )
    assert any(
        row["parameter"] == "learning_rate" and row["value"] == "0.0001" for row in rows
    )
    assert any(
        row["parameter"] == "batch_size" and row["value"] == "64" for row in rows
    )
