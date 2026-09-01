from scripts.finalize_g2_geometry_offline import bootstrap_mean_ci, describe, quantile


def test_describe_and_quantile() -> None:
    assert quantile([0.0, 1.0, 2.0], 0.5) == 1.0
    report = describe([1.0, 2.0, 3.0])
    assert report["known"] == 3
    assert report["mean_eV_per_atom"] == 2.0
    assert report["median_eV_per_atom"] == 2.0


def test_bootstrap_is_deterministic_and_directional() -> None:
    left = bootstrap_mean_ci([-2.0, -1.0, -0.5], seed=7, replicates=200)
    right = bootstrap_mean_ci([-2.0, -1.0, -0.5], seed=7, replicates=200)
    assert left == right
    assert left[1] < 0
