from __future__ import annotations

import argparse
from pathlib import Path

from .paper import (
    ARCH_SWEEP_CONFIGS,
    DEFAULT_GRID,
    PAPER_ROOT,
    SWITCH_STUDY_CONFIGS,
    evaluate_manifest,
    plot_results,
    run_smoke,
    train_config_set,
    write_arch_sweep_summary,
)
from .runtime import display_path, parse_int_list, project_root


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible training job.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="artifacts/runs/manual")
    parser.add_argument("--run-slug", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    args = parser.parse_args()

    from .paper import run_training_job

    config = project_root() / args.config
    label = args.run_slug or Path(args.config).stem
    run_dir = run_training_job(
        config=config,
        label=label,
        output_root=project_root() / args.output_root,
        seed=args.seed,
        device=args.device,
        wandb_enabled=args.wandb,
    )
    print(display_path(run_dir))


def eval_ab_main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one run on the A/B switch grid."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--direction",
        choices=["linear_to_quadratic", "quadratic_to_linear"],
        required=True,
    )
    parser.add_argument("--a-values", default=DEFAULT_GRID)
    parser.add_argument("--b-values", default=DEFAULT_GRID)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from .paper import evaluate_manifest
    from .runtime import write_json

    manifest_path = project_root() / args.output_dir / "single_run_manifest.json"
    write_json(
        manifest_path,
        {
            "kind": "single-run",
            "runs": [
                {
                    "label": Path(args.run_dir).name,
                    "run_dir": args.run_dir,
                }
            ],
        },
    )
    evaluate_manifest(
        manifest_path,
        results_dir=project_root() / args.output_dir,
        directions=[args.direction],
        a_values=parse_int_list(args.a_values),
        b_values=parse_int_list(args.b_values),
        trials=args.trials,
        batch_size=args.batch_size,
        step=args.step,
        device=args.device,
    )


def paper_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper reproduction workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    arch_parser = subparsers.add_parser("arch-sweep")
    arch_parser.add_argument("--seed", type=int, default=0)
    arch_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    arch_parser.add_argument("--wandb", action="store_true")

    train_parser = subparsers.add_parser("train-curricula")
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    train_parser.add_argument("--wandb", action="store_true")

    eval_parser = subparsers.add_parser("eval-switch")
    eval_parser.add_argument(
        "--manifest",
        default=str(PAPER_ROOT / "run_manifest.json"),
    )
    eval_parser.add_argument("--a-values", default=DEFAULT_GRID)
    eval_parser.add_argument("--b-values", default=DEFAULT_GRID)
    eval_parser.add_argument("--trials", type=int, default=1000)
    eval_parser.add_argument("--batch-size", type=int, default=None)
    eval_parser.add_argument("--step", type=int, default=-1)
    eval_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    plot_parser = subparsers.add_parser("plot-paper")
    plot_parser.add_argument("--results-dir", default=str(PAPER_ROOT / "results"))

    all_parser = subparsers.add_parser("all")
    all_parser.add_argument("--seed", type=int, default=0)
    all_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    all_parser.add_argument("--wandb", action="store_true")
    all_parser.add_argument("--trials", type=int, default=1000)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--output-root", default="artifacts")
    smoke_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
    )

    args = parser.parse_args()
    if args.command == "arch-sweep":
        manifest = train_config_set(
            ARCH_SWEEP_CONFIGS,
            output_root=project_root() / "artifacts" / "runs" / "paper" / "arch_sweep",
            manifest_path=PAPER_ROOT / "arch_sweep" / "run_manifest.json",
            seed=args.seed,
            device=args.device,
            wandb_enabled=args.wandb,
            kind="arch-sweep",
        )
        summary_csv = write_arch_sweep_summary(
            manifest,
            PAPER_ROOT / "arch_sweep" / "summary.csv",
        )
        print(summary_csv.relative_to(project_root()))
        return

    if args.command == "train-curricula":
        manifest = train_config_set(
            SWITCH_STUDY_CONFIGS,
            output_root=(
                project_root() / "artifacts" / "runs" / "paper" / "switch_study"
            ),
            manifest_path=PAPER_ROOT / "run_manifest.json",
            seed=args.seed,
            device=args.device,
            wandb_enabled=args.wandb,
            kind="switch-study",
        )
        print(manifest.relative_to(project_root()))
        return

    if args.command == "eval-switch":
        results_dir = evaluate_manifest(
            project_root() / args.manifest,
            results_dir=PAPER_ROOT / "results",
            directions=["linear_to_quadratic", "quadratic_to_linear"],
            a_values=parse_int_list(args.a_values),
            b_values=parse_int_list(args.b_values),
            trials=args.trials,
            batch_size=args.batch_size,
            step=args.step,
            device=args.device,
        )
        print(results_dir.relative_to(project_root()))
        return

    if args.command == "plot-paper":
        figures_dir = plot_results(
            project_root() / args.results_dir,
            PAPER_ROOT / "figures",
            directions=["linear_to_quadratic", "quadratic_to_linear"],
        )
        print(figures_dir.relative_to(project_root()))
        return

    if args.command == "all":
        train_config_set(
            ARCH_SWEEP_CONFIGS,
            output_root=project_root() / "artifacts" / "runs" / "paper" / "arch_sweep",
            manifest_path=PAPER_ROOT / "arch_sweep" / "run_manifest.json",
            seed=args.seed,
            device=args.device,
            wandb_enabled=args.wandb,
            kind="arch-sweep",
        )
        train_config_set(
            SWITCH_STUDY_CONFIGS,
            output_root=(
                project_root() / "artifacts" / "runs" / "paper" / "switch_study"
            ),
            manifest_path=PAPER_ROOT / "run_manifest.json",
            seed=args.seed,
            device=args.device,
            wandb_enabled=args.wandb,
            kind="switch-study",
        )
        evaluate_manifest(
            PAPER_ROOT / "run_manifest.json",
            results_dir=PAPER_ROOT / "results",
            directions=["linear_to_quadratic", "quadratic_to_linear"],
            a_values=parse_int_list(DEFAULT_GRID),
            b_values=parse_int_list(DEFAULT_GRID),
            trials=args.trials,
            batch_size=None,
            step=-1,
            device=args.device,
        )
        figures_dir = plot_results(
            PAPER_ROOT / "results",
            PAPER_ROOT / "figures",
            directions=["linear_to_quadratic", "quadratic_to_linear"],
        )
        print(figures_dir.relative_to(project_root()))
        return

    if args.command == "smoke":
        smoke_root = run_smoke(project_root() / args.output_root, args.device)
        print(smoke_root.relative_to(project_root()))
