# CS182 ICL Reproduction

This repository reproduces the experiments in [ICL_extended-8.pdf](./ICL_extended-8.pdf), our CS182 paper on context stickiness in in-context learning for linear and quadratic regression tasks.

The maintained workflow is:

1. `uv` for environment management
2. script-first experiment entrypoints
3. deterministic configs under `configs/`
4. generated outputs under `artifacts/`
5. pytest + CI smoke checks so the repo does not drift into “works on my machine”

`eval.ipynb` and the plotting notebooks are still useful for exploration, but the CLI below is the authoritative reproduction path.

![](setting.jpg)

## Quick Start

```bash
uv sync --extra dev
uv run pytest
uv run icl-paper smoke
```

The smoke command trains tiny CPU models for all three curricula, runs a tiny A/B evaluation, and writes manifests, CSVs, and plots under `artifacts/smoke/`.

## Reproducing `ICL_extended-8.pdf`

### Architecture Sweep

Section 6 uses three linear-only architecture configs.

```bash
uv run icl-paper arch-sweep --device cuda
```

Outputs:

- `artifacts/paper/icl_extended/arch_sweep/run_manifest.json`
- `artifacts/paper/icl_extended/arch_sweep/summary.csv`

### Train the Three Curricula

```bash
uv run icl-paper train-curricula --device cuda
```

Output:

- `artifacts/paper/icl_extended/run_manifest.json`

### Evaluate Both Switch Directions

```bash
uv run icl-paper eval-switch --device cuda --trials 1000
```

Outputs:

- `artifacts/paper/icl_extended/results/*_linear_to_quadratic_mean.csv`
- `artifacts/paper/icl_extended/results/*_linear_to_quadratic_sem.csv`
- `artifacts/paper/icl_extended/results/*_quadratic_to_linear_mean.csv`
- `artifacts/paper/icl_extended/results/*_quadratic_to_linear_sem.csv`

### Regenerate Figures

```bash
uv run icl-paper plot-paper
```

Outputs:

- `artifacts/paper/icl_extended/figures/*.png`

### End-to-End

```bash
uv run icl-paper all --device cuda --trials 1000
```

## Single-Run Commands

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
- `src/cs182_project/` packaged CLI and orchestration code
- `src/` core model, training, task, and evaluation logic
- `artifacts/` generated runs, CSVs, figures, and manifests
- `docs/icl_extended.md` figure-to-command mapping for the paper
- `tests/` unit tests and smoke coverage

## Reproducibility Notes

- `uv` with Python 3.10 is the only supported environment path.
- W&B is optional and disabled by default in the supported configs.
- Each run writes `config.yaml`, `history.jsonl`, `state.pt`, and `summary.json`.
- The supported paper workflow is from scratch only; it does not rely on hosted checkpoints.

## Citation

```bibtex
@InProceedings{garg2022what,
  title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
  author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
  year={2022},
  booktitle={arXiv preprint}
}
```
