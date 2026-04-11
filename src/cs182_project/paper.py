from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from .plotting import plot_heatmap, plot_recovery_curves
from .runtime import (
    discover_run_dir,
    display_path,
    ensure_legacy_src_on_path,
    parse_int_list,
    project_root,
    run_command,
    timestamp_utc,
    write_json,
    write_overlay_config,
)

DEFAULT_GRID = "0,1,2,3,4,6,8,10,12,14,16,18,20"
SMOKE_GRID = "0,2"
PAPER_ROOT = project_root() / "artifacts" / "paper" / "icl_extended"


def config_path(*parts: str) -> Path:
    return project_root().joinpath("configs", *parts)


ARCH_SWEEP_CONFIGS = [
    ("small", config_path("paper", "arch_sweep", "linear_small.yaml")),
    ("medium", config_path("paper", "arch_sweep", "linear_medium.yaml")),
    ("large", config_path("paper", "arch_sweep", "linear_large.yaml")),
]

SWITCH_STUDY_CONFIGS = [
    ("dual_random", config_path("paper", "switch_study", "dual_random.yaml")),
    ("dual_mixed", config_path("paper", "switch_study", "dual_mixed.yaml")),
    ("dual_sequential", config_path("paper", "switch_study", "dual_sequential.yaml")),
]

SMOKE_CONFIGS = [
    ("dual_random", config_path("smoke", "dual_random.yaml")),
    ("dual_mixed", config_path("smoke", "dual_mixed.yaml")),
    ("dual_sequential", config_path("smoke", "dual_sequential.yaml")),
]


def run_training_job(
    *,
    config: Path,
    label: str,
    output_root: Path,
    seed: int,
    device: str,
    wandb_enabled: bool,
) -> Path:
    base_dir = output_root / label
    base_dir.mkdir(parents=True, exist_ok=True)
    before = {path.name for path in base_dir.iterdir() if path.is_dir()}
    overlay_config = write_overlay_config(
        config,
        {
            "seed": seed,
            "device": device,
            "compute_metrics_on_finish": False,
            "out_dir": str(base_dir),
            "wandb": {"enabled": wandb_enabled},
        },
    )
    env = {"WANDB_MODE": "disabled"} if not wandb_enabled else {}
    try:
        run_command(
            [sys.executable, "src/train.py", "--config", str(overlay_config)],
            env=env,
        )
    finally:
        overlay_config.unlink(missing_ok=True)
    return discover_run_dir(base_dir, before)


def train_config_set(
    configs: list[tuple[str, Path]],
    *,
    output_root: Path,
    manifest_path: Path,
    seed: int,
    device: str,
    wandb_enabled: bool,
    kind: str,
) -> Path:
    runs = []
    for label, cfg_path in configs:
        run_dir = run_training_job(
            config=cfg_path,
            label=label,
            output_root=output_root,
            seed=seed,
            device=device,
            wandb_enabled=wandb_enabled,
        )
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "label": label,
                "config": str(cfg_path.relative_to(project_root())),
                "run_dir": display_path(run_dir),
                "summary": summary,
            }
        )
    payload = {
        "kind": kind,
        "created_at": timestamp_utc(),
        "device": device,
        "seed": seed,
        "runs": runs,
    }
    write_json(manifest_path, payload)
    return manifest_path


def write_arch_sweep_summary(manifest_path: Path, output_csv: Path) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "run_dir", "final_loss", "step", "device", "seed"],
        )
        writer.writeheader()
        for run in manifest["runs"]:
            writer.writerow(
                {
                    "label": run["label"],
                    "run_dir": run["run_dir"],
                    "final_loss": run["summary"].get("overall_loss"),
                    "step": run["summary"].get("step"),
                    "device": run["summary"].get("device"),
                    "seed": run["summary"].get("seed"),
                }
            )
    return output_csv


def evaluate_manifest(
    manifest_path: Path,
    *,
    results_dir: Path,
    directions: list[str],
    a_values: list[int],
    b_values: list[int],
    trials: int,
    batch_size: int | None,
    step: int,
    device: str,
) -> Path:
    ensure_legacy_src_on_path()
    from dual_eval_2 import run_dual_eval, run_dual_eval_switched

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results_dir.mkdir(parents=True, exist_ok=True)
    for run in manifest["runs"]:
        run_dir = Path(run["run_dir"])
        if not run_dir.is_absolute():
            run_dir = project_root() / run_dir
        label = run["label"]
        if "linear_to_quadratic" in directions:
            mean_df, sem_df = run_dual_eval(
                run_dir=str(run_dir),
                a_examples=a_values,
                b_examples=b_values,
                trials=trials,
                batch_size=batch_size,
                step=step,
                device=device,
            )
            mean_df.to_csv(results_dir / f"{label}_linear_to_quadratic_mean.csv")
            sem_df.to_csv(results_dir / f"{label}_linear_to_quadratic_sem.csv")
        if "quadratic_to_linear" in directions:
            mean_df, sem_df = run_dual_eval_switched(
                run_dir=str(run_dir),
                a_examples=a_values,
                b_examples=b_values,
                trials=trials,
                batch_size=batch_size,
                step=step,
                device=device,
            )
            mean_df.to_csv(results_dir / f"{label}_quadratic_to_linear_mean.csv")
            sem_df.to_csv(results_dir / f"{label}_quadratic_to_linear_sem.csv")
    return results_dir


def plot_results(results_dir: Path, figures_dir: Path, directions: list[str]) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in SWITCH_STUDY_CONFIGS]
    for direction in directions:
        mean_paths = []
        present_labels = []
        for label in labels:
            csv_path = results_dir / f"{label}_{direction}_mean.csv"
            if not csv_path.exists():
                continue
            mean_paths.append(csv_path)
            present_labels.append(label)
            plot_heatmap(
                csv_path,
                title=f"{label} {direction.replace('_', ' ')}",
                output_path=figures_dir / f"{label}_{direction}_heatmap.png",
            )
        if mean_paths:
            plot_recovery_curves(
                mean_paths,
                present_labels,
                figures_dir / f"{direction}_recovery_curves.png",
                fixed_pre_counts=[0, 2],
            )
    return figures_dir


def run_smoke(output_root: Path, device: str) -> Path:
    smoke_root = output_root / "smoke"
    manifest_path = smoke_root / "run_manifest.json"
    train_config_set(
        SMOKE_CONFIGS,
        output_root=smoke_root / "runs",
        manifest_path=manifest_path,
        seed=0,
        device=device,
        wandb_enabled=False,
        kind="smoke",
    )
    evaluate_manifest(
        manifest_path,
        results_dir=smoke_root / "results",
        directions=["linear_to_quadratic", "quadratic_to_linear"],
        a_values=parse_int_list(SMOKE_GRID),
        b_values=parse_int_list(SMOKE_GRID),
        trials=2,
        batch_size=2,
        step=-1,
        device=device,
    )
    plot_results(
        smoke_root / "results",
        smoke_root / "figures",
        directions=["linear_to_quadratic", "quadratic_to_linear"],
    )
    return smoke_root
