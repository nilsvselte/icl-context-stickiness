# `ICL_extended-8.pdf` Reproduction Map

This document maps each supported paper artifact to the command that regenerates it.

## Environment Check

```bash
uv sync --extra dev
uv run pytest
uv run icl-paper smoke
```

## Section 6: Architecture Sweep

```bash
uv run icl-paper arch-sweep --device cuda
```

Outputs:

- `artifacts/paper/icl_extended/arch_sweep/run_manifest.json`
- `artifacts/paper/icl_extended/arch_sweep/summary.csv`

Configs:

- `configs/paper/arch_sweep/linear_small.yaml`
- `configs/paper/arch_sweep/linear_medium.yaml`
- `configs/paper/arch_sweep/linear_large.yaml`

## Sections 5 and 7: Switch-Study Training

```bash
uv run icl-paper train-curricula --device cuda
```

Output:

- `artifacts/paper/icl_extended/run_manifest.json`

Configs:

- `configs/paper/switch_study/dual_random.yaml`
- `configs/paper/switch_study/dual_mixed.yaml`
- `configs/paper/switch_study/dual_sequential.yaml`

## Section 5.3: A/B Evaluation Grid

```bash
uv run icl-paper eval-switch --device cuda --trials 1000
```

Grid:

- `n_pre ∈ {0,1,2,3,4,6,8,10,12,14,16,18,20}`
- `n_post ∈ {0,1,2,3,4,6,8,10,12,14,16,18,20}`

Outputs:

- `artifacts/paper/icl_extended/results/dual_random_linear_to_quadratic_mean.csv`
- `artifacts/paper/icl_extended/results/dual_random_linear_to_quadratic_sem.csv`
- `artifacts/paper/icl_extended/results/dual_random_quadratic_to_linear_mean.csv`
- same pattern for `dual_mixed` and `dual_sequential`

## Figure Regeneration

```bash
uv run icl-paper plot-paper
```

Outputs:

- per-model heatmaps for both directions under `artifacts/paper/icl_extended/figures/`
- direction-level recovery curves under `artifacts/paper/icl_extended/figures/`

## Single-Run Debugging

```bash
uv run icl-train --config configs/paper/switch_study/dual_random.yaml --device cpu
```

```bash
uv run icl-eval-ab \
  --run-dir artifacts/runs/manual/dual_random/<run-id> \
  --direction linear_to_quadratic \
  --output-dir artifacts/manual_eval
```
