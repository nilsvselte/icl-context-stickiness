# When Context Sticks Reproduction Map

This document maps each maintained artifact for *When Context Sticks: Studying Interference in In-Context Learning* to the command that regenerates it.

The setup is based on the synthetic regression framework introduced in:

Shivam Garg, Dimitris Tsipras, Percy Liang, and Gregory Valiant. *What Can Transformers Learn In-Context? A Case Study of Simple Function Classes.* 2022.

The commands below are organized around our paper’s tables and figures.

## Prior Work Citation

If you need to cite the setup this repository builds on, use:

```bibtex
@article{garg2022what,
  title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
  author={Garg, Shivam and Tsipras, Dimitris and Liang, Percy and Valiant, Gregory},
  year={2022}
}
```

## Environment Check

```bash
uv sync --extra dev
uv run pytest
uv run icl-paper smoke --device cpu
```

## Table 1: Architecture Sweep

```bash
uv run icl-paper arch-sweep --device cuda
```

Outputs:

- `artifacts/paper/when_context_sticks/arch_sweep/run_manifest.json`
- `artifacts/paper/when_context_sticks/table_1_architecture_sweep.csv`

Configs:

- `configs/paper/arch_sweep/linear_small.yaml`
- `configs/paper/arch_sweep/linear_medium.yaml`
- `configs/paper/arch_sweep/linear_large.yaml`

Notes:

- The table records both average loss over training and final loss.
- The average loss column is the paper-facing model-selection metric.

## Table 2: Main Hyperparameters

```bash
uv run icl-paper arch-sweep --device cuda
```

Output:

- `artifacts/paper/when_context_sticks/table_2_hyperparameters.csv`

Primary sources:

- `configs/fragments/model_medium.yaml`
- `configs/fragments/training_dual_paper.yaml`

## Train the Three Curricula

```bash
uv run icl-paper train-curricula --device cuda
```

Output:

- `artifacts/paper/when_context_sticks/run_manifest.json`

Configs:

- `configs/paper/switch_study/dual_random.yaml`
- `configs/paper/switch_study/dual_mixed.yaml`
- `configs/paper/switch_study/dual_sequential.yaml`

## Evaluate the A/B Switch Grid

```bash
uv run icl-paper eval-switch --device cuda --trials 1000
```

Grid:

- `n_pre in {0,1,2,3,4,6,8,10,12,14,16,18,20}`
- `n_post in {0,1,2,3,4,6,8,10,12,14,16,18,20}`

Outputs:

- `artifacts/paper/when_context_sticks/results/dual_random_linear_to_quadratic_mean.csv`
- `artifacts/paper/when_context_sticks/results/dual_random_linear_to_quadratic_sem.csv`
- `artifacts/paper/when_context_sticks/results/dual_random_quadratic_to_linear_mean.csv`
- the same filename pattern for `dual_mixed` and `dual_sequential`

## Figures 1-6

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

Figure mapping:

- Figure 1 uses the full mean-surface CSVs for all three curricula and both switch directions.
- Figure 2 uses SEM-aware recovery curves for fixed pre-switch counts in the linear-to-quadratic direction.
- Figure 3 uses SEM-aware stickiness curves for fixed post-switch counts in the linear-to-quadratic direction.
- Figure 4 uses the sequential curriculum only, plotting recovery as quadratic examples accumulate after the switch.
- Figure 5 compares the two switch directions for the sequential curriculum across fixed first-context counts.
- Figure 6 compares mixed and random curricula across both switch directions.

## End-to-End Command

```bash
uv run icl-paper all --device cuda --trials 1000
```

This is the full supported paper reproduction command.

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
