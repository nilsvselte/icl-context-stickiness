from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRICULUM_LABELS = {
    "dual_random": "Random",
    "dual_mixed": "Mixed",
    "dual_sequential": "Sequential",
}

CURRICULUM_COLORS = {
    "dual_random": "#4C78A8",
    "dual_mixed": "#72B7B2",
    "dual_sequential": "#E45756",
}

DIRECTION_LABELS = {
    "linear_to_quadratic": "Linear -> Quadratic",
    "quadratic_to_linear": "Quadratic -> Linear",
}

DIRECTION_COLORS = {
    "linear_to_quadratic": "#F58518",
    "quadratic_to_linear": "#54A24B",
}

DIRECTION_MARKERS = {
    "linear_to_quadratic": "o",
    "quadratic_to_linear": "s",
}


def load_result_frame(csv_path: Path) -> tuple[str, list[int], list[int], pd.DataFrame]:
    frame = pd.read_csv(csv_path)
    index_name = frame.columns[0]
    value_frame = frame.set_index(index_name)
    value_frame.index = [int(value) for value in value_frame.index]
    value_frame.columns = [
        int(str(column).split("_")[-1]) for column in value_frame.columns
    ]
    return (
        index_name,
        value_frame.index.tolist(),
        value_frame.columns.tolist(),
        value_frame,
    )


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    *,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_surface_grid(
    result_paths: dict[str, dict[str, Path]],
    output_path: Path,
    *,
    labels: list[str],
    directions: list[str],
) -> None:
    fig = plt.figure(figsize=(18, 10))
    plot_index = 1
    for row_direction in directions:
        for label in labels:
            ax = fig.add_subplot(
                len(directions), len(labels), plot_index, projection="3d"
            )
            plot_index += 1
            csv_path = result_paths.get(label, {}).get(row_direction)
            if csv_path is None or not csv_path.exists():
                ax.set_axis_off()
                continue
            index_name, pre_counts, post_counts, frame = load_result_frame(csv_path)
            x_grid, y_grid = np.meshgrid(post_counts, pre_counts)
            ax.plot_surface(
                x_grid,
                y_grid,
                frame.values,
                cmap="viridis",
                linewidth=0,
                antialiased=False,
                alpha=0.95,
            )
            ax.set_title(
                (
                    f"{CURRICULUM_LABELS.get(label, label)}\n"
                    f"{DIRECTION_LABELS[row_direction]}"
                ),
                fontsize=10,
            )
            ax.set_xlabel("Post-switch examples")
            ax.set_ylabel(index_name.replace("_", " "))
            ax.set_zlabel("MSE")
    fig.suptitle(
        "Figure 1. Overall error surfaces across curricula and switch directions"
    )
    save_figure(fig, output_path, rect=(0, 0, 1, 0.96))


def plot_recovery_grid(
    mean_paths: dict[str, Path],
    sem_paths: dict[str, Path],
    output_path: Path,
    *,
    fixed_pre_counts: list[int],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    handles = []
    labels_for_legend = []
    for axis, pre_count in zip(axes.flat, fixed_pre_counts):
        plotted = False
        for label, csv_path in mean_paths.items():
            if not csv_path.exists():
                continue
            _, pre_values, post_counts, mean_frame = load_result_frame(csv_path)
            if pre_count not in pre_values:
                continue
            _, _, _, sem_frame = load_result_frame(sem_paths[label])
            handle = axis.errorbar(
                post_counts,
                mean_frame.loc[pre_count].values,
                yerr=sem_frame.loc[pre_count].values,
                color=CURRICULUM_COLORS[label],
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=CURRICULUM_LABELS[label],
            )
            plotted = True
            if label not in labels_for_legend:
                handles.append(handle)
                labels_for_legend.append(label)
        axis.set_title(f"Pre-switch examples = {pre_count}")
        axis.set_xlabel("Post-switch quadratic examples")
        axis.set_ylabel("Mean squared error")
        if not plotted:
            axis.text(0.5, 0.5, "Not available", ha="center", va="center")
            axis.set_axis_off()
    fig.suptitle("Figure 2. Recovery after switching from linear to quadratic tasks")
    if handles:
        fig.legend(
            handles,
            [CURRICULUM_LABELS[label] for label in labels_for_legend],
            loc="upper center",
            ncol=len(handles),
            frameon=False,
        )
    save_figure(fig, output_path, rect=(0, 0, 1, 0.93))


def plot_stickiness_grid(
    mean_paths: dict[str, Path],
    sem_paths: dict[str, Path],
    output_path: Path,
    *,
    fixed_post_counts: list[int],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    handles = []
    labels_for_legend = []
    for axis, post_count in zip(axes.flat, fixed_post_counts):
        plotted = False
        for label, csv_path in mean_paths.items():
            if not csv_path.exists():
                continue
            _, pre_counts, post_values, mean_frame = load_result_frame(csv_path)
            if post_count not in post_values:
                continue
            _, _, _, sem_frame = load_result_frame(sem_paths[label])
            handle = axis.errorbar(
                pre_counts,
                mean_frame[post_count].values,
                yerr=sem_frame[post_count].values,
                color=CURRICULUM_COLORS[label],
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=CURRICULUM_LABELS[label],
            )
            plotted = True
            if label not in labels_for_legend:
                handles.append(handle)
                labels_for_legend.append(label)
        axis.set_title(f"Post-switch examples = {post_count}")
        axis.set_xlabel("Pre-switch linear examples")
        axis.set_ylabel("Mean squared error")
        if not plotted:
            axis.text(0.5, 0.5, "Not available", ha="center", va="center")
            axis.set_axis_off()
    fig.suptitle("Figure 3. Stickiness as a function of pre-switch exposure")
    if handles:
        fig.legend(
            handles,
            [CURRICULUM_LABELS[label] for label in labels_for_legend],
            loc="upper center",
            ncol=len(handles),
            frameon=False,
        )
    save_figure(fig, output_path, rect=(0, 0, 1, 0.93))


def plot_sequential_recovery(
    mean_path: Path,
    sem_path: Path,
    output_path: Path,
    *,
    fixed_pre_counts: list[int],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    _, pre_counts, post_counts, mean_frame = load_result_frame(mean_path)
    _, _, _, sem_frame = load_result_frame(sem_path)
    for pre_count in fixed_pre_counts:
        if pre_count not in pre_counts:
            continue
        ax.errorbar(
            post_counts,
            mean_frame.loc[pre_count].values,
            yerr=sem_frame.loc[pre_count].values,
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=f"Pre-switch linear examples = {pre_count}",
        )
    ax.set_title(
        "Figure 4. Sequential curriculum recovery across post-switch quadratic examples"
    )
    ax.set_xlabel("Post-switch quadratic examples")
    ax.set_ylabel("Mean squared error")
    ax.legend(fontsize=8, frameon=False)
    save_figure(fig, output_path)


def plot_direction_comparison_grid(
    direction_paths: dict[str, tuple[Path, Path]],
    output_path: Path,
    *,
    fixed_first_context_counts: list[int],
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    handles = []
    legend_labels = []
    for axis, count in zip(axes.flat, fixed_first_context_counts):
        plotted = False
        for direction, (mean_path, sem_path) in direction_paths.items():
            _, first_counts, second_counts, mean_frame = load_result_frame(mean_path)
            if count not in first_counts:
                continue
            _, _, _, sem_frame = load_result_frame(sem_path)
            handle = axis.errorbar(
                second_counts,
                mean_frame.loc[count].values,
                yerr=sem_frame.loc[count].values,
                marker=DIRECTION_MARKERS[direction],
                color=DIRECTION_COLORS[direction],
                linewidth=1.8,
                capsize=3,
                label=DIRECTION_LABELS[direction],
            )
            plotted = True
            if direction not in legend_labels:
                handles.append(handle)
                legend_labels.append(direction)
        axis.set_title(f"First-context examples = {count}")
        axis.set_xlabel("Second-context examples")
        axis.set_ylabel("Mean squared error")
        if not plotted:
            axis.text(0.5, 0.5, "Not available", ha="center", va="center")
            axis.set_axis_off()
    fig.suptitle(title)
    if handles:
        fig.legend(
            handles,
            [DIRECTION_LABELS[label] for label in legend_labels],
            loc="upper center",
            ncol=len(handles),
            frameon=False,
        )
    save_figure(fig, output_path, rect=(0, 0, 1, 0.93))


def plot_curriculum_direction_grid(
    result_paths: dict[str, dict[str, Path]],
    sem_paths: dict[str, dict[str, Path]],
    output_path: Path,
    *,
    fixed_first_context_counts: list[int],
    curricula: list[str],
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    handles = []
    legend_labels = []
    line_styles = {"dual_random": "--", "dual_mixed": "-"}
    for axis, count in zip(axes.flat, fixed_first_context_counts):
        plotted = False
        for curriculum in curricula:
            for direction in DIRECTION_LABELS:
                mean_path = result_paths.get(curriculum, {}).get(direction)
                sem_path = sem_paths.get(curriculum, {}).get(direction)
                if mean_path is None or sem_path is None or not mean_path.exists():
                    continue
                _, first_counts, second_counts, mean_frame = load_result_frame(
                    mean_path
                )
                if count not in first_counts:
                    continue
                _, _, _, sem_frame = load_result_frame(sem_path)
                legend_label = (
                    f"{CURRICULUM_LABELS[curriculum]} | {DIRECTION_LABELS[direction]}"
                )
                handle = axis.errorbar(
                    second_counts,
                    mean_frame.loc[count].values,
                    yerr=sem_frame.loc[count].values,
                    marker=DIRECTION_MARKERS[direction],
                    color=CURRICULUM_COLORS[curriculum],
                    linestyle=line_styles[curriculum],
                    linewidth=1.8,
                    capsize=3,
                    label=legend_label,
                )
                plotted = True
                if legend_label not in legend_labels:
                    handles.append(handle)
                    legend_labels.append(legend_label)
        axis.set_title(f"First-context examples = {count}")
        axis.set_xlabel("Second-context examples")
        axis.set_ylabel("Mean squared error")
        if not plotted:
            axis.text(0.5, 0.5, "Not available", ha="center", va="center")
            axis.set_axis_off()
    fig.suptitle(title)
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            ncol=len(handles),
            frameon=False,
        )
    save_figure(fig, output_path, rect=(0, 0, 1, 0.93))
