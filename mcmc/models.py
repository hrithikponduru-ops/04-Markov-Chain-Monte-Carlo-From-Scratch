"""Log densities of the targets sampled in this project.

Every function returns log p(theta) up to an additive constant. Metropolis
only ever looks at differences log p(theta') - log p(theta), so constants
cancel and normalising integrals never have to be computed.
"""

from dataclasses import dataclass

import numpy as np


def log_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """log N(x; mean, cov) including its normalising constant, for d >= 1."""
    x, mean, cov = np.atleast_1d(x), np.atleast_1d(mean), np.atleast_2d(cov)
    d = x.size
    diff = x - mean
    quadratic = float(diff @ np.linalg.solve(cov, diff))
    _, log_det = np.linalg.slogdet(cov)
    return -0.5 * (quadratic + log_det + d * np.log(2 * np.pi))


def log_beta_binomial_posterior(theta: float, k: int, n: int, a: float, b: float) -> float:
    """Unnormalised log posterior for a coin's heads probability theta.

    Prior Beta(a, b), likelihood Binomial(n, theta) with k heads:
    |   p(theta | k) ∝ theta^(a + k - 1) (1 - theta)^(b + n - k - 1),
    which is the Beta(a + k, b + n - k) density up to a constant.
    Outside (0, 1) the density is zero, so the log density is -inf.
    """
    theta = float(theta)
    if not 0.0 < theta < 1.0:
        return -np.inf
    return (a + k - 1) * np.log(theta) + (b + n - k - 1) * np.log(1 - theta)


@dataclass(frozen=True)
class RegressionPriors:
    """Independent zero-mean Gaussian priors on slope and intercept.

    The noise scale sigma gets a flat prior on log sigma, which is the
    scale-invariant choice: it says nothing about whether sigma is nearer
    1 or 1000. Working with log sigma also lets the random walk roam over
    the whole real line without ever proposing a negative sigma.
    """

    slope_sd: float
    intercept_sd: float


def log_linear_regression_posterior(
    params: np.ndarray, x: np.ndarray, y: np.ndarray, priors: RegressionPriors
) -> float:
    """log p(slope, intercept, log_sigma | x, y) up to a constant.

    Model: y_i = intercept + slope * x_i + eps_i, eps_i ~ N(0, sigma^2).
    log likelihood = -n log sigma - sum(residuals^2) / (2 sigma^2)  (+ const)
    log prior      = -slope^2 / (2 sd_s^2) - intercept^2 / (2 sd_i^2) (+ const)
    """
    slope, intercept, log_sigma = params
    sigma = np.exp(log_sigma)
    residuals = y - intercept - slope * x
    log_likelihood = -len(y) * log_sigma - float(residuals @ residuals) / (2 * sigma**2)
    log_prior = -(slope**2) / (2 * priors.slope_sd**2) - (intercept**2) / (2 * priors.intercept_sd**2)
    return log_likelihood + log_prior