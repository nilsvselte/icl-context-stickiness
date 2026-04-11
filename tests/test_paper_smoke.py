from pathlib import Path

from when_context_sticks.paper import run_smoke


def test_smoke_pipeline_generates_manifest_results_and_figures(tmp_path: Path):
    smoke_root = run_smoke(tmp_path, device="cpu")
    assert (smoke_root / "run_manifest.json").exists()
    assert (
        smoke_root / "results" / "dual_random_linear_to_quadratic_mean.csv"
    ).exists()
    assert (
        smoke_root / "results" / "dual_sequential_quadratic_to_linear_mean.csv"
    ).exists()
    assert (smoke_root / "figures" / "figure_1_overall_3d_error_surfaces.png").exists()
    assert (smoke_root / "figures" / "figure_2_recovery_curves.png").exists()
    assert (smoke_root / "figures" / "figure_3_stickiness_curves.png").exists()
    assert (
        smoke_root / "figures" / "figure_4_sequential_error_vs_quadratic_examples.png"
    ).exists()
    assert (
        smoke_root / "figures" / "figure_5_sequential_switch_comparison.png"
    ).exists()
    assert (
        smoke_root / "figures" / "figure_6_mixed_random_switch_comparison.png"
    ).exists()
