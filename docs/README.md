# Repository Guide

The supported workflow now lives in:

- `README.md` for the main setup and reproduction commands
- `docs/icl_extended.md` for the figure-by-figure paper mapping
- `configs/` for maintained experiment specs
- `src/cs182_project/` for the packaged CLI/orchestration layer

The research core remains in `src/`:

- `src/train.py` training loop and checkpointing
- `src/dual_eval_2.py` A/B context-switch evaluation
- `src/models.py`, `src/tasks.py`, `src/samplers.py` model and data definitions
- `src/schema.py` Quinine validation schema

Generated outputs should go under `artifacts/`, not `src/results/` or ad hoc run folders.

## Maintained Commands

- `uv run icl-train --config <config>`
- `uv run icl-eval-ab --run-dir <run-dir> --direction <direction> --output-dir <dir>`
- `uv run icl-paper arch-sweep`
- `uv run icl-paper train-curricula`
- `uv run icl-paper eval-switch`
- `uv run icl-paper plot-paper`
- `uv run icl-paper smoke`

## Development Notes

- Add new experiment YAMLs under `configs/`, not `src/conf/`.
- Keep `wandb.enabled: false` by default for reproducible configs.
- Prefer extending the packaged CLI in `src/cs182_project/` rather than adding new shell scripts.
- Add tests or smoke coverage before relying on a new training/evaluation path.
