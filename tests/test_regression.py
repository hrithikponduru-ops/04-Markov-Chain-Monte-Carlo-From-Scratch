"""Bayesian linear regression with a nearly flat prior must agree with
ordinary least squares: the posterior mean of (slope, intercept) is the OLS
solution, and the MCMC estimate of it differs only by Monte Carlo error.
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcmc.diagnostics import effective_sample_size  # noqa: E402
from mcmc.metropolis import metropolis_hastings  # noqa: E402
from mcmc.models import RegressionPriors, log_gaussian, log_linear_regression_posterior  # noqa: E402


def _synthetic():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 50)
    y = 2.0 + 3.0 * x + rng.normal(scale=0.5, size=50)
    return x, y


def test_posterior_mean_matches_ols():
    x, y = _synthetic()
    priors = RegressionPriors(slope_sd=100.0, intercept_sd=100.0)
    ols_slope, ols_intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]

    rng = np.random.default_rng(1)
    chain = metropolis_hastings(
        lambda p: log_linear_regression_posterior(p, x, y, priors),
        np.array([ols_slope, ols_intercept, np.log(0.5)]),
        np.array([0.02, 0.12, 0.12]), 60_000, rng,
    )
    samples = chain.samples[5_000:]
    for j, ols in enumerate((ols_slope, ols_intercept)):
        col = samples[:, j]
        standard_error = col.std() / np.sqrt(effective_sample_size(col))
        assert abs(col.mean() - ols) < 4.0 * standard_error


def test_posterior_sigma_is_near_truth():
    x, y = _synthetic()
    priors = RegressionPriors(slope_sd=100.0, intercept_sd=100.0)
    rng = np.random.default_rng(2)
    chain = metropolis_hastings(
        lambda p: log_linear_regression_posterior(p, x, y, priors),
        np.array([3.0, 2.0, 0.0]), np.array([0.02, 0.12, 0.12]), 60_000, rng,
    )
    sigma = np.exp(chain.samples[5_000:, 2])
    # With 50 points the posterior sd of sigma is roughly sigma / sqrt(2 n) ~ 0.05.
    assert abs(sigma.mean() - 0.5) < 0.15


def test_log_gaussian_matches_scipy():
    mean = np.array([1.0, -2.0])
    cov = np.array([[2.0, 0.3], [0.3, 1.0]])
    x = np.array([0.5, -1.0])
    assert np.isclose(log_gaussian(x, mean, cov), stats.multivariate_normal(mean, cov).logpdf(x))


def test_regression_log_posterior_matches_direct_formula():
    x, y = _synthetic()
    priors = RegressionPriors(slope_sd=10.0, intercept_sd=10.0)
    slope, intercept, log_sigma = 3.1, 1.9, np.log(0.6)
    sigma = np.exp(log_sigma)
    residuals = y - intercept - slope * x
    expected = (
        stats.norm(0, sigma).logpdf(residuals).sum()
        + stats.norm(0, 10.0).logpdf(slope)
        + stats.norm(0, 10.0).logpdf(intercept)
    )
    mine = log_linear_regression_posterior(np.array([slope, intercept, log_sigma]), x, y, priors)
    # Both may drop different additive constants; compare differences between two points.
    slope2 = 2.9
    expected2 = (
        stats.norm(0, sigma).logpdf(y - intercept - slope2 * x).sum()
        + stats.norm(0, 10.0).logpdf(slope2)
        + stats.norm(0, 10.0).logpdf(intercept)
    )
    mine2 = log_linear_regression_posterior(np.array([slope2, intercept, log_sigma]), x, y, priors)
    assert np.isclose(mine - mine2, expected - expected2, atol=1e-9)