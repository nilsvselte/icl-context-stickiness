from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_heatmap(csv_path: Path, title: str, output_path: Path) -> None:
    frame = pd.read_csv(csv_path, index_col=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(frame.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Post-switch examples")
    ax.set_ylabel("Pre-switch examples")
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels([column.split("_")[-1] for column in frame.columns], rotation=45)
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels(frame.index.tolist())
    fig.colorbar(image, ax=ax, label="Mean squared error")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_recovery_curves(
    csv_paths: list[Path],
    labels: list[str],
    output_path: Path,
    fixed_pre_counts: list[int],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for csv_path, label in zip(csv_paths, labels):
        frame = pd.read_csv(csv_path, index_col=0)
        for pre_count in fixed_pre_counts:
            if pre_count not in frame.index:
                continue
            x_values = [int(column.split("_")[-1]) for column in frame.columns]
            ax.plot(
                x_values,
                frame.loc[pre_count].values,
                marker="o",
                label=f"{label} pre={pre_count}",
            )
    ax.set_xlabel("Post-switch examples")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Recovery curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
