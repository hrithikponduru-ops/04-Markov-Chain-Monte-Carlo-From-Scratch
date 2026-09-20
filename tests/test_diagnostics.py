"""Convergence diagnostics checked on processes with known autocorrelation.

An AR(1) process x_t = rho x_{t-1} + eps_t has autocorrelation rho^k at lag k,
so its integrated autocorrelation time is 1 + 2 sum_k rho^k = (1 + rho) / (1 - rho).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcmc.diagnostics import (  # noqa: E402
    autocorrelation,
    effective_sample_size,
    gelman_rubin_rhat,
    integrated_autocorrelation_time,
)

def _ar1(rho, n, rng):
    eps = rng.normal(size=n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1 - rho**2)  # start in the stationary distribution
    for t in range(1, n):
        x[t] = rho * x[t - 1] + eps[t]
    return x

def test_white_noise_has_no_autocorrelation_and_full_ess():
    n = 100_000
    x = np.random.default_rng(0).normal(size=n)
    rho = autocorrelation(x, max_lag=20)
    assert rho[0] == 1.0
    assert np.all(np.abs(rho[1:]) < 4.0 / np.sqrt(n))
    assert abs(effective_sample_size(x) / n - 1.0) < 0.1

def test_ar1_integrated_autocorrelation_time():
    rho = 0.9
    x = _ar1(rho, 400_000, np.random.default_rng(1))
    tau_true = (1 + rho) / (1 - rho)  # = 19
    tau = integrated_autocorrelation_time(x)
    assert abs(tau / tau_true - 1.0) < 0.1
    assert abs(effective_sample_size(x) - len(x) / tau) < 1e-9

def test_ar1_autocorrelation_decays_geometrically():
    rho = 0.7
    x = _ar1(rho, 400_000, np.random.default_rng(2))
    est = autocorrelation(x, max_lag=5)
    assert np.allclose(est, rho ** np.arange(6), atol=0.02)

def test_rhat_is_one_for_chains_from_the_same_distribution():
    rng = np.random.default_rng(3)
    chains = rng.normal(size=(4, 5_000))
    assert abs(gelman_rubin_rhat(chains) - 1.0) < 0.01

def test_rhat_flags_chains_that_disagree():
    rng = np.random.default_rng(4)
    chains = rng.normal(size=(4, 5_000)) + np.array([[0.0], [1.0], [2.0], [3.0]])
    assert gelman_rubin_rhat(chains) > 1.1