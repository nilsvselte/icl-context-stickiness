from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import yaml

from .plotting import (
    plot_curriculum_direction_grid,
    plot_direction_comparison_grid,
    plot_recovery_grid,
    plot_sequential_recovery,
    plot_stickiness_grid,
    plot_surface_grid,
)
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
RECOVERY_COUNTS = [0, 2, 4, 8, 12, 18]
STICKINESS_COUNTS = [2, 4, 6, 8, 12, 18]
DIRECTION_COMPARISON_COUNTS = [0, 2, 4, 8, 12, 16]
PAPER_ROOT = project_root() / "artifacts" / "paper" / "when_context_sticks"


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


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_run_dir(run_dir: str) -> Path:
    path = Path(run_dir)
    if path.is_absolute():
        return path
    return project_root() / path


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
            fieldnames=[
                "label",
                "n_embd",
                "n_layer",
                "n_head",
                "average_loss",
                "average_excess_loss",
                "final_loss",
                "step",
                "run_dir",
                "device",
                "seed",
            ],
        )
        writer.writeheader()
        for run in manifest["runs"]:
            run_dir = resolve_run_dir(run["run_dir"])
            config = load_yaml(run_dir / "config.yaml")
            model = config.get("model", {})
            summary = run["summary"]
            writer.writerow(
                {
                    "label": run["label"],
                    "n_embd": model.get("n_embd"),
                    "n_layer": model.get("n_layer"),
                    "n_head": model.get("n_head"),
                    "average_loss": summary.get("average_loss"),
                    "average_excess_loss": summary.get("average_excess_loss"),
                    "final_loss": summary.get("overall_loss"),
                    "step": summary.get("step"),
                    "run_dir": run["run_dir"],
                    "device": summary.get("device"),
                    "seed": summary.get("seed"),
                }
            )
    return output_csv


def write_hyperparameter_table(output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    training = load_yaml(config_path("fragments", "training_dual_paper.yaml"))
    model = load_yaml(config_path("fragments", "model_medium.yaml"))
    rows = [
        {
            "section": "setup",
            "parameter": "base_experimental_setup",
            "value": (
                "Garg et al. (2022), What Can Transformers Learn In-Context? "
                "A Case Study of Simple Function Classes"
            ),
            "source": "paper framing",
        },
        {
            "section": "model",
            "parameter": "family",
            "value": model["model"]["family"],
            "source": "configs/fragments/model_medium.yaml",
        },
        {
            "section": "model",
            "parameter": "n_dims",
            "value": model["model"]["n_dims"],
            "source": "configs/fragments/model_medium.yaml",
        },
        {
            "section": "model",
            "parameter": "n_positions",
            "value": model["model"]["n_positions"],
            "source": "configs/fragments/model_medium.yaml",
        },
        {
            "section": "model",
            "parameter": "n_embd",
            "value": model["model"]["n_embd"],
            "source": "configs/fragments/model_medium.yaml",
        },
        {
            "section": "model",
            "parameter": "n_layer",
            "value": model["model"]["n_layer"],
            "source": "configs/fragments/model_medium.yaml",
        },
        {
            "section": "model",
            "parameter": "n_head",
            "value": model["model"]["n_head"],
            "source": "configs/fragments/model_medium.yaml",
        },
        {
            "section": "training",
            "parameter": "optimizer",
            "value": "Adam",
            "source": "src/train.py",
        },
        {
            "section": "training",
            "parameter": "learning_rate",
            "value": training["training"]["learning_rate"],
            "source": "configs/fragments/training_dual_paper.yaml",
        },
        {
            "section": "training",
            "parameter": "batch_size",
            "value": training["training"]["batch_size"],
            "source": "configs/fragments/training_dual_paper.yaml",
        },
        {
            "section": "training",
            "parameter": "train_steps",
            "value": training["training"]["train_steps"],
            "source": "configs/fragments/training_dual_paper.yaml",
        },
        {
            "section": "training",
            "parameter": "points_start",
            "value": training["training"]["curriculum"]["points"]["start"],
            "source": "configs/fragments/training_dual_paper.yaml",
        },
        {
            "section": "training",
            "parameter": "points_end",
            "value": training["training"]["curriculum"]["points"]["end"],
            "source": "configs/fragments/training_dual_paper.yaml",
        },
        {
            "section": "training",
            "parameter": "points_inc",
            "value": training["training"]["curriculum"]["points"]["inc"],
            "source": "configs/fragments/training_dual_paper.yaml",
        },
        {
            "section": "training",
            "parameter": "points_interval",
            "value": training["training"]["curriculum"]["points"]["interval"],
            "source": "configs/fragments/training_dual_paper.yaml",
        },
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["section", "parameter", "value", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)
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
        run_dir = resolve_run_dir(run["run_dir"])
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


def collect_result_paths(
    results_dir: Path,
    labels: list[str],
    directions: list[str],
) -> tuple[dict[str, dict[str, Path]], dict[str, dict[str, Path]]]:
    mean_paths: dict[str, dict[str, Path]] = {label: {} for label in labels}
    sem_paths: dict[str, dict[str, Path]] = {label: {} for label in labels}
    for label in labels:
        for direction in directions:
            mean_paths[label][direction] = results_dir / f"{label}_{direction}_mean.csv"
            sem_paths[label][direction] = results_dir / f"{label}_{direction}_sem.csv"
    return mean_paths, sem_paths


def plot_results(results_dir: Path, figures_dir: Path, directions: list[str]) -> Path:
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _ in SWITCH_STUDY_CONFIGS]
    mean_paths, sem_paths = collect_result_paths(results_dir, labels, directions)

    plot_surface_grid(
        mean_paths,
        figures_dir / "figure_1_overall_3d_error_surfaces.png",
        labels=labels,
        directions=directions,
    )

    linear_to_quadratic_means = {
        label: mean_paths[label]["linear_to_quadratic"]
        for label in labels
        if "linear_to_quadratic" in mean_paths[label]
    }
    linear_to_quadratic_sems = {
        label: sem_paths[label]["linear_to_quadratic"]
        for label in labels
        if "linear_to_quadratic" in sem_paths[label]
    }
    plot_recovery_grid(
        linear_to_quadratic_means,
        linear_to_quadratic_sems,
        figures_dir / "figure_2_recovery_curves.png",
        fixed_pre_counts=RECOVERY_COUNTS,
    )
    plot_stickiness_grid(
        linear_to_quadratic_means,
        linear_to_quadratic_sems,
        figures_dir / "figure_3_stickiness_curves.png",
        fixed_post_counts=STICKINESS_COUNTS,
    )

    plot_sequential_recovery(
        mean_paths["dual_sequential"]["linear_to_quadratic"],
        sem_paths["dual_sequential"]["linear_to_quadratic"],
        figures_dir / "figure_4_sequential_error_vs_quadratic_examples.png",
        fixed_pre_counts=RECOVERY_COUNTS,
    )

    plot_direction_comparison_grid(
        {
            direction: (
                mean_paths["dual_sequential"][direction],
                sem_paths["dual_sequential"][direction],
            )
            for direction in directions
        },
        figures_dir / "figure_5_sequential_switch_comparison.png",
        fixed_first_context_counts=DIRECTION_COMPARISON_COUNTS,
        title="Figure 5. Sequential curriculum comparison across switch directions",
    )

    plot_curriculum_direction_grid(
        mean_paths,
        sem_paths,
        figures_dir / "figure_6_mixed_random_switch_comparison.png",
        fixed_first_context_counts=DIRECTION_COMPARISON_COUNTS,
        curricula=["dual_mixed", "dual_random"],
        title=(
            "Figure 6. Mixed and random curriculum comparison across switch directions"
        ),
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
