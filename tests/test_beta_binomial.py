"""The coin problem has a closed-form posterior, so MCMC can be checked exactly.

Prior Beta(a, b), k heads in n flips -> posterior Beta(a + k, b + n - k).
With a = b = 2, k = 7, n = 10 the posterior is Beta(9, 5). SciPy supplies the
reference distribution; the sampler never sees it.
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcmc.diagnostics import effective_sample_size  # noqa: E402
from mcmc.metropolis import metropolis_hastings  # noqa: E402
from mcmc.models import log_beta_binomial_posterior  # noqa: E402

K, N, A, B = 7, 10, 2.0, 2.0
POSTERIOR = stats.beta(A + K, B + N - K)

def _coin_chain():
    rng = np.random.default_rng(0)
    chain = metropolis_hastings(
        lambda t: log_beta_binomial_posterior(t[0], K, N, A, B),
        np.array([0.5]), 0.25, 100_000, rng,
    )
    return chain.samples[2_000:, 0]

def test_posterior_mean_matches_closed_form():
    x = _coin_chain()
    n_eff = effective_sample_size(x)
    standard_error = POSTERIOR.std() / np.sqrt(n_eff)
    assert abs(x.mean() - POSTERIOR.mean()) < 4.0 * standard_error

def test_posterior_variance_matches_closed_form():
    x = _coin_chain()
    n_eff = effective_sample_size(x)
    true_var = POSTERIOR.var()
    # Relative error of a variance estimate is about sqrt(2 / n_eff) for
    # near-Gaussian shapes; the Beta(9, 5) is skewed, so allow 6 standard errors.
    assert abs(x.var() - true_var) < 6.0 * true_var * np.sqrt(2.0 / n_eff)

def test_posterior_quantiles_match_closed_form():
    x = _coin_chain()
    n_eff = effective_sample_size(x)
    for p in (0.05, 0.50, 0.95):
        q_true = POSTERIOR.ppf(p)
        # Standard error of a sample quantile: sqrt(p(1-p)/n) / f(q).
        standard_error = np.sqrt(p * (1 - p) / n_eff) / POSTERIOR.pdf(q_true)
        assert abs(np.quantile(x, p) - q_true) < 4.0 * standard_error

def test_log_posterior_is_minus_infinity_outside_unit_interval():
    assert log_beta_binomial_posterior(-0.1, K, N, A, B) == -np.inf
    assert log_beta_binomial_posterior(1.1, K, N, A, B) == -np.inf
    assert np.isfinite(log_beta_binomial_posterior(0.5, K, N, A, B))

def test_log_posterior_matches_scipy_up_to_a_constant():
    thetas = np.array([0.2, 0.5, 0.7, 0.9])
    mine = np.array([log_beta_binomial_posterior(t, K, N, A, B) for t in thetas])
    ref = POSTERIOR.logpdf(thetas)
    diffs = mine - ref
    assert np.allclose(diffs, diffs[0], atol=1e-12)