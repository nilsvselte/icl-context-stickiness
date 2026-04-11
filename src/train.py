import json
import os
import random
import uuid
from random import randint

import numpy as np
import torch
import wandb
import yaml
from quinine import QuinineArgumentParser
from tqdm import tqdm

from curriculum import Curriculum
from eval import get_run_metrics
from models import build_model
from samplers import get_data_sampler
from schema import schema
from task_labeling import TaskLabeler
from tasks import get_task_sampler

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cpu")


def resolve_device(device_name):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_history(history_path, payload):
    with open(history_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args, task_labeler):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)
    history_path = os.path.join(args.out_dir, "history.jsonl")

    starting_step = 0
    loss_total = 0.0
    loss_count = 0
    excess_loss_total = 0.0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        loss_total = state.get("loss_total", 0.0)
        loss_count = state.get("loss_count", 0)
        excess_loss_total = state.get("excess_loss_total", 0.0)
        for i in range(state["train_step"] + 1):
            curriculum.update()

    feature_dims = task_labeler.feature_dims
    total_dims = task_labeler.model_n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=total_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        feature_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples
    last_summary = None
    wandb_enabled = bool(getattr(args.wandb, "enabled", True)) and not args.test_run

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        truncation = task_labeler.augmentation_truncation(curriculum.n_dims_truncated)
        xs = data_sampler.sample_xs(
            curriculum.n_points, bsize, truncation, **data_sampler_args
        )
        task = task_sampler(**task_sampler_args)
        feature_xs = task_labeler.feature_slice(xs)
        if task_labeler.enabled:
            ys, metadata = task.evaluate(feature_xs, return_metadata=True)
            label_name = metadata.get("task_label", task.task_label)
            xs = task_labeler.apply(xs, label_name)
        else:
            ys = task.evaluate(feature_xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(
            model, xs.to(device), ys.to(device), optimizer, loss_func
        )

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        log_payload = {
            "step": i,
            "overall_loss": loss,
            "excess_loss": loss / baseline_loss,
            "n_points": curriculum.n_points,
            "n_dims": curriculum.n_dims_truncated,
        }
        loss_total += loss
        excess_loss_total += log_payload["excess_loss"]
        loss_count += 1
        should_log = (
            i % args.wandb.log_every_steps == 0 or i == args.training.train_steps - 1
        )
        if should_log:
            append_history(history_path, log_payload)
            if wandb_enabled:
                wandb.log(
                    {
                        **log_payload,
                        "pointwise/loss": dict(
                            zip(point_wise_tags, point_wise_loss.cpu().numpy())
                        ),
                    },
                    step=i,
                )
        last_summary = log_payload

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
                "loss_total": loss_total,
                "loss_count": loss_count,
                "excess_loss_total": excess_loss_total,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))

    summary = last_summary or {
        "step": starting_step,
        "overall_loss": None,
        "n_points": curriculum.n_points,
        "n_dims": curriculum.n_dims_truncated,
    }
    summary["average_loss"] = loss_total / loss_count if loss_count else None
    summary["average_excess_loss"] = (
        excess_loss_total / loss_count if loss_count else None
    )
    return summary


def train_dual_curriculum(model, args, task_labeler):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)
    history_path = os.path.join(args.out_dir, "history.jsonl")
    starting_step = 0
    loss_total = 0.0
    loss_count = 0
    excess_loss_total = 0.0
    state_path = os.path.join(args.out_dir, "state.pt")
    mode = args.training.curriculum_type

    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        loss_total = state.get("loss_total", 0.0)
        loss_count = state.get("loss_count", 0)
        excess_loss_total = state.get("excess_loss_total", 0.0)
        for i in range(state["train_step"] + 1):
            curriculum.update()

    feature_dims = task_labeler.feature_dims
    total_dims = task_labeler.model_n_dims
    bsize = args.training.batch_size
    feature_dims = task_labeler.feature_dims
    total_dims = task_labeler.model_n_dims
    bsize = args.training.batch_size

    data_sampler = get_data_sampler(args.training.data, n_dims=total_dims)

    linear_sampler = get_task_sampler(
        "linear_regression",
        feature_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )

    quadratic_sampler = get_task_sampler(
        "quadratic_regression",
        feature_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )

    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples
    last_summary = None
    wandb_enabled = bool(getattr(args.wandb, "enabled", True)) and not args.test_run

    for i in pbar:
        if mode == "sequential":
            if i < args.training.train_steps // 2:
                task_sampler = linear_sampler
            else:
                task_sampler = quadratic_sampler

        elif mode == "random":
            _ = random.random()
            if _ < 0.5:
                task_sampler = linear_sampler
            else:
                task_sampler = quadratic_sampler

        elif mode == "mixed":
            if i < args.training.train_steps // 2:
                task_sampler = linear_sampler
            else:
                _ = random.random()
                if _ < 0.5:
                    task_sampler = linear_sampler
                else:
                    task_sampler = quadratic_sampler

        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        truncation = task_labeler.augmentation_truncation(curriculum.n_dims_truncated)
        xs = data_sampler.sample_xs(
            curriculum.n_points, bsize, truncation, **data_sampler_args
        )
        task = task_sampler(**task_sampler_args)
        feature_xs = task_labeler.feature_slice(xs)
        if task_labeler.enabled:
            ys, metadata = task.evaluate(feature_xs, return_metadata=True)
            label_name = metadata.get("task_label", task.task_label)
            xs = task_labeler.apply(xs, label_name)
        else:
            ys = task.evaluate(feature_xs)

        loss_func = task.get_training_metric()

        loss, output = train_step(
            model, xs.to(device), ys.to(device), optimizer, loss_func
        )

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        log_payload = {
            "step": i,
            "overall_loss": loss,
            "excess_loss": loss / baseline_loss,
            "n_points": curriculum.n_points,
            "n_dims": curriculum.n_dims_truncated,
            "curriculum_type": mode,
        }
        loss_total += loss
        excess_loss_total += log_payload["excess_loss"]
        loss_count += 1
        should_log = (
            i % args.wandb.log_every_steps == 0 or i == args.training.train_steps - 1
        )
        if should_log:
            append_history(history_path, log_payload)
            if wandb_enabled:
                wandb.log(
                    {
                        **log_payload,
                        "pointwise/loss": dict(
                            zip(point_wise_tags, point_wise_loss.cpu().numpy())
                        ),
                    },
                    step=i,
                )
        last_summary = log_payload

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
                "loss_total": loss_total,
                "loss_count": loss_count,
                "excess_loss_total": excess_loss_total,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))

    summary = last_summary or {
        "step": starting_step,
        "overall_loss": None,
        "n_points": curriculum.n_points,
        "n_dims": curriculum.n_dims_truncated,
        "curriculum_type": mode,
    }
    summary["average_loss"] = loss_total / loss_count if loss_count else None
    summary["average_excess_loss"] = (
        excess_loss_total / loss_count if loss_count else None
    )
    return summary


def main(args, task_labeler):
    global device
    set_random_seed(args.seed)
    device = resolve_device(getattr(args, "device", "auto"))
    if args.test_run:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    if bool(getattr(args.wandb, "enabled", True)) and not args.test_run:
        wandb_project = os.getenv("WANDB_PROJECT") or args.wandb.project
        wandb_entity = os.getenv("WANDB_ENTITY") or args.wandb.entity
        if wandb_entity in {"", "local"}:
            wandb_entity = None
        wandb.init(
            dir=args.out_dir,
            project=wandb_project,
            entity=wandb_entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,
        )

    model = build_model(args.model)
    model.to(device)
    model.train()

    if args.training.problem_type is not None:
        summary = train_dual_curriculum(model, args, task_labeler)
    else:
        summary = train(model, args, task_labeler)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                **summary,
                "device": str(device),
                "seed": args.seed,
                "run_name": args.wandb.name,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    if getattr(args, "compute_metrics_on_finish", False) and not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    task_labeler = TaskLabeler(
        getattr(args.training, "task_labeling", None), args.model.n_dims
    )
    args.model.n_dims = task_labeler.model_n_dims

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args, task_labeler)
