from pathlib import Path

from cs182_project.paper import run_smoke


def test_smoke_pipeline_generates_manifest_results_and_figures(tmp_path: Path):
    smoke_root = run_smoke(tmp_path, device="cpu")
    assert (smoke_root / "run_manifest.json").exists()
    assert (
        smoke_root / "results" / "dual_random_linear_to_quadratic_mean.csv"
    ).exists()
    assert (
        smoke_root / "results" / "dual_sequential_quadratic_to_linear_mean.csv"
    ).exists()
    assert (smoke_root / "figures" / "linear_to_quadratic_recovery_curves.png").exists()
