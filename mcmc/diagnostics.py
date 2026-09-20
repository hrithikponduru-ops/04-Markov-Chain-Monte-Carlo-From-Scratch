"""Convergence diagnostics: how many independent samples is a chain worth?

Successive MCMC samples are correlated, so n samples carry less information
than n independent draws. The integrated autocorrelation time tau says how
many steps it takes for the chain to forget where it was; the effective
sample size is n / tau.
"""

import numpy as np

SOKAL_WINDOW_FACTOR = 5.0

def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """rho_k = sum_t (x_t - m)(x_{t+k} - m) / sum_t (x_t - m)^2 for k = 0..max_lag.

    The denominator is the same for every lag (the "biased" estimator). It
    makes rho_0 = 1 exactly and keeps the sequence positive semi-definite,
    which the running sum in integrated_autocorrelation_time relies on.
    """
    xc = np.asarray(x, dtype=float) - np.mean(x)
    denominator = float(xc @ xc)
    rho = np.array([float(xc[:len(xc) - k] @ xc[k:]) for k in range(max_lag + 1)])
    return rho / denominator

def integrated_autocorrelation_time(x: np.ndarray, max_lag: int | None = None) -> float:
    """tau = 1 + 2 sum_{k=1}^{M} rho_k, with the window M chosen automatically.

    Summing all lags is useless: past a few times tau the true rho_k is zero
    and the estimates are pure noise, which adds a random walk to the sum.
    Sokal's rule stops at the first M with M >= c * tau_M (c = 5): large
    enough to capture the decay, small enough to exclude most of the noise.
    """
    n = len(x)
    if max_lag is None:
        max_lag = min(n - 1, 1_000)
    rho = autocorrelation(x, max_lag)
    running_tau = 1.0 + 2.0 * np.cumsum(rho[1:])  # running_tau[M-1] = tau using lags 1..M
    lags = np.arange(1, max_lag + 1)
    window = np.flatnonzero(lags >= SOKAL_WINDOW_FACTOR * running_tau)
    tau = running_tau[window[0]] if window.size else running_tau[-1]
    return float(max(tau, 1.0))

def effective_sample_size(x: np.ndarray, max_lag: int | None = None) -> float:
    """n_eff = n / tau: the number of independent draws the chain is worth."""
    return len(x) / integrated_autocorrelation_time(x, max_lag)

def gelman_rubin_rhat(chains: np.ndarray) -> float:
    """Potential scale reduction factor for m chains of length n (shape (m, n)).

    Compare the variance between chain means (B) with the variance within
    each chain (W). If all chains explore the same distribution, B is small
    and R-hat is near 1. If they are stuck in different places, B is large
    and R-hat exceeds 1. The (n-1)/n and 1/n weights come from the unbiased
    estimate of the pooled variance.
    """
    chains = np.asarray(chains, dtype=float)
    m, n = chains.shape
    if m < 2:
        raise ValueError("R-hat needs at least two chains")
    chain_means = chains.mean(axis=1)
    within = chains.var(axis=1, ddof=1).mean()  # W
    between = n * chain_means.var(ddof=1)  # B
    pooled = (n - 1) / n * within + between / n
    return float(np.sqrt(pooled / within))