from bootstrap_metrics import paired_bootstrap_rate_ci


def test_paired_bootstrap_rate_ci_recovers_positive_delta():
    base_flags = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0] * 20
    other_flags = [1, 1, 0, 1, 1, 0, 1, 0, 1, 1] * 20
    result = paired_bootstrap_rate_ci(base_flags, other_flags, n_bootstrap=1000, seed=1)
    assert result["observed_delta"] > 0
    assert result["delta_ci95"][0] > 0
    assert result["fraction_delta_positive"] > 0.95


def test_paired_bootstrap_rate_ci_rejects_length_mismatch():
    import pytest
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_rate_ci([1, 0], [1], n_bootstrap=10, seed=0)


def test_interval_key_names_track_the_confidence_level():
    base_flags = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0] * 20
    other_flags = [1, 1, 0, 1, 1, 0, 1, 0, 1, 1] * 20
    result = paired_bootstrap_rate_ci(base_flags, other_flags, n_bootstrap=200, seed=3, alpha=0.01)
    assert "delta_ci99" in result
    assert "base_ci99" in result
    assert "other_ci99" in result
    assert "delta_ci95" not in result
