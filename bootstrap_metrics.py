"""Paired bootstrap confidence intervals for binary exact-arg deltas."""

from __future__ import annotations

import random


def paired_bootstrap_rate_ci(base_flags, other_flags, n_bootstrap=5000, seed=42, alpha=0.05):
    if len(base_flags) != len(other_flags):
        raise ValueError("base_flags and other_flags must have the same length")
    if not base_flags:
        raise ValueError("flags must be nonempty")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = len(base_flags)
    base_rate = sum(base_flags) / n
    other_rate = sum(other_flags) / n
    observed_delta = other_rate - base_rate
    rng = random.Random(seed)
    deltas = []
    base_rates = []
    other_rates = []
    for _ in range(n_bootstrap):
        indices = [rng.randrange(n) for _ in range(n)]
        base_sample = sum(base_flags[i] for i in indices) / n
        other_sample = sum(other_flags[i] for i in indices) / n
        base_rates.append(base_sample)
        other_rates.append(other_sample)
        deltas.append(other_sample - base_sample)
    deltas.sort()
    base_rates.sort()
    other_rates.sort()
    lower_index = int((alpha / 2) * n_bootstrap)
    upper_index = int((1 - alpha / 2) * n_bootstrap) - 1
    interval_suffix = f"ci{int(round((1 - alpha) * 100))}"
    return {
        "n_examples": n,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "observed_base": base_rate,
        "observed_other": other_rate,
        "observed_delta": observed_delta,
        f"delta_{interval_suffix}": [deltas[lower_index], deltas[upper_index]],
        f"base_{interval_suffix}": [base_rates[lower_index], base_rates[upper_index]],
        f"other_{interval_suffix}": [other_rates[lower_index], other_rates[upper_index]],
        "fraction_delta_positive": sum(delta > 0 for delta in deltas) / n_bootstrap,
    }
