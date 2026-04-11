# When Context Sticks: Studying Interference in In-Context Learning

This repository is the maintained reproduction entrypoint for our paper *When Context Sticks: Studying Interference in In-Context Learning*.

The experiments build on the synthetic linear and quadratic regression setup introduced in:

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes.* 2022.

The configs, commands, tables, and figures here are organized around our paper and its context-switching results.

The supported workflow is:

1. `uv` for environment management
2. script-first experiment entrypoints
3. deterministic configs under `configs/`
4. generated outputs under `artifacts/`
5. tests plus CI smoke coverage so the repo stays reproducible on a fresh clone

The CLI below is the authoritative reproduction path.

## Quick Start

```bash
uv sync --extra dev
uv run pytest
uv run icl-paper smoke --device cpu
```

The smoke workflow trains tiny CPU models for all three curricula, runs a tiny A/B evaluation grid, and writes manifests, CSVs, and paper-style figures under `artifacts/smoke/`.

## Full Reproduction

Full paper-scale reproduction is a GPU workflow. The supported artifact root is `artifacts/paper/when_context_sticks/`.

### Table 1 and Table 2

```bash
uv run icl-paper arch-sweep --device cuda
```

Outputs:

- `artifacts/paper/when_context_sticks/arch_sweep/run_manifest.json`
- `artifacts/paper/when_context_sticks/table_1_architecture_sweep.csv`
- `artifacts/paper/when_context_sticks/table_2_hyperparameters.csv`

`table_1_architecture_sweep.csv` reports the architecture sweep with average loss over training and the final loss for each model size.

### Train the Three Curricula

```bash
uv run icl-paper train-curricula --device cuda
```

Output:

- `artifacts/paper/when_context_sticks/run_manifest.json`

### Evaluate Both Switch Directions

```bash
uv run icl-paper eval-switch --device cuda --trials 1000
```

Outputs:

- `artifacts/paper/when_context_sticks/results/*_linear_to_quadratic_mean.csv`
- `artifacts/paper/when_context_sticks/results/*_linear_to_quadratic_sem.csv`
- `artifacts/paper/when_context_sticks/results/*_quadratic_to_linear_mean.csv`
- `artifacts/paper/when_context_sticks/results/*_quadratic_to_linear_sem.csv`

### Regenerate the Paper Figures

```bash
uv run icl-paper plot-paper
```

Outputs:

- `artifacts/paper/when_context_sticks/figures/figure_1_overall_3d_error_surfaces.png`
- `artifacts/paper/when_context_sticks/figures/figure_2_recovery_curves.png`
- `artifacts/paper/when_context_sticks/figures/figure_3_stickiness_curves.png`
- `artifacts/paper/when_context_sticks/figures/figure_4_sequential_error_vs_quadratic_examples.png`
- `artifacts/paper/when_context_sticks/figures/figure_5_sequential_switch_comparison.png`
- `artifacts/paper/when_context_sticks/figures/figure_6_mixed_random_switch_comparison.png`

### End to End

```bash
uv run icl-paper all --device cuda --trials 1000
```

This runs the architecture sweep, writes both paper tables, trains the three curricula, evaluates both switch directions, and regenerates all six paper figures.

## Single-Run Debugging

Train one config:

```bash
uv run icl-train --config configs/paper/switch_study/dual_random.yaml --device cuda
```

Evaluate one run directory on one switch direction:

```bash
uv run icl-eval-ab \
  --run-dir artifacts/runs/paper/switch_study/dual_random/<run-id> \
  --direction linear_to_quadratic \
  --output-dir artifacts/manual_eval
```

## Repository Layout

- `configs/` maintained experiment configs
- `src/when_context_sticks/` packaged CLI and paper orchestration
- `src/` core model, training, task, and evaluation logic
- `artifacts/` generated runs, CSVs, figures, and manifests
- `docs/when_context_sticks.md` table-and-figure reproduction map
- `tests/` unit tests and smoke coverage

## Reproducibility Notes

- `uv` with Python 3.10 is the only supported environment path.
- The supported GPU path uses the pinned `torch==2.5.1` dependency so `uv sync --extra dev` works on current CUDA GPUs such as NVIDIA L4.
- W&B is optional and disabled by default in the maintained configs.
- `WANDB_PROJECT` and `WANDB_ENTITY` can be used to override the default online logging destination without editing configs.
- Each run writes `config.yaml`, `history.jsonl`, `state.pt`, and `summary.json`.
- The architecture sweep summary uses average loss over training so the reported table matches the intended paper metric.
- The supported workflow is from scratch only; it does not depend on pre-hosted checkpoints.

## Citation Basis

This repository reuses the experimental setup introduced by Garg et al. for synthetic in-context linear and quadratic regression. If you describe the provenance of the setup, cite that paper directly:

```bibtex
@article{garg2022what,
  title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
  author={Garg, Shivam and Tsipras, Dimitris and Liang, Percy and Valiant, Gregory},
  year={2022}
}
```
