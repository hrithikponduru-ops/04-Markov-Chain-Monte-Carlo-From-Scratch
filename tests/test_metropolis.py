"""Verify the Metropolis-Hastings sampler on targets whose answers are known.

Each test encodes a mathematical fact that must hold if the derivation in
README.md is correct. Tolerances come from the Monte Carlo standard error
sigma / sqrt(n_eff), not from magic numbers.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcmc.diagnostics import effective_sample_size  # noqa: E402
from mcmc.metropolis import metropolis_hastings  # noqa: E402
from mcmc.models import log_gaussian  # noqa: E402

def _standard_normal(theta):
    return log_gaussian(theta, np.zeros(1), np.eye(1))

def test_standard_gaussian_mean_and_variance():
    rng = np.random.default_rng(0)
    chain = metropolis_hastings(_standard_normal, np.zeros(1), 2.4, 50_000, rng)
    x = chain.samples[1_000:, 0]
    n_eff = effective_sample_size(x)
    # Monte Carlo error of the mean is sigma / sqrt(n_eff); 4 standard errors.
    assert abs(x.mean()) < 4.0 / np.sqrt(n_eff)
    # Variance of a sample variance estimate for a Gaussian is 2 sigma^4 / n_eff.
    assert abs(x.var() - 1.0) < 4.0 * np.sqrt(2.0 / n_eff)

def test_correlated_2d_gaussian_covariance():
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    rng = np.random.default_rng(1)
    chain = metropolis_hastings(
        lambda t: log_gaussian(t, np.zeros(2), cov), np.zeros(2), 1.5, 200_000, rng
    )
    sample_cov = np.cov(chain.samples[2_000:].T)
    assert np.allclose(sample_cov, cov, rtol=0.05, atol=0.03)

def test_acceptance_rate_for_tuned_step_is_sensible():
    # For a 1-D Gaussian target the optimal random-walk step is about 2.4 sigma,
    # which gives an acceptance rate near 0.44 (Roberts, Gelman & Gilks 1997).
    rng = np.random.default_rng(2)
    chain = metropolis_hastings(_standard_normal, np.zeros(1), 2.4, 20_000, rng)
    assert 0.35 < chain.acceptance_rate < 0.55

def test_chain_is_deterministic_for_fixed_seed():
    a = metropolis_hastings(_standard_normal, np.zeros(1), 1.0, 500, np.random.default_rng(7))
    b = metropolis_hastings(_standard_normal, np.zeros(1), 1.0, 500, np.random.default_rng(7))
    assert np.array_equal(a.samples, b.samples)
    assert np.array_equal(a.accepted, b.accepted)

def test_chain_shapes_and_first_sample():
    rng = np.random.default_rng(3)
    chain = metropolis_hastings(_standard_normal, np.array([0.3]), 1.0, 100, rng)
    assert chain.samples.shape == (100, 1)
    assert chain.log_probs.shape == (100,)
    assert chain.accepted.shape == (100,)
    assert chain.accepted.dtype == bool
    # Rejected steps repeat the previous sample exactly.
    for i in range(1, 100):
        if not chain.accepted[i]:
            assert np.array_equal(chain.samples[i], chain.samples[i - 1])

def test_starting_outside_support_raises():
    def log_target(theta):
        return -np.inf if theta[0] < 0 else -theta[0]

    with pytest.raises(ValueError):
        metropolis_hastings(log_target, np.array([-1.0]), 1.0, 10, np.random.default_rng(0))