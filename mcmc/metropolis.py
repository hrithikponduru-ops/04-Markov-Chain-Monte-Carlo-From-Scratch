"""The Metropolis-Hastings algorithm with a Gaussian random-walk proposal.

Everything is done in log space. Posterior densities for even modest data
sets are products of many small numbers and underflow to 0.0 in floating
point; their logarithms are perfectly ordinary numbers like -340.2.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

LogDensity = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class StepResult:
    """Everything that happened in one Metropolis-Hastings step."""

    proposal: np.ndarray
    log_p_proposal: float
    log_alpha: float  # log of the acceptance probability, <= 0
    uniform: float  # the U(0, 1) draw compared against alpha
    accepted: bool
    theta: np.ndarray  # state after the step
    log_p: float  # its log density


@dataclass(frozen=True)
class Chain:
    samples: np.ndarray  # shape (n_steps, d)
    log_probs: np.ndarray  # shape (n_steps,)
    accepted: np.ndarray  # shape (n_steps,), bool

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted.mean())


def metropolis_step(
    log_target: LogDensity,
    theta: np.ndarray,
    log_p: float,
    step_size: float | np.ndarray,
    rng: np.random.Generator,
    log_proposal_ratio: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> StepResult:
    """One step. Propose theta' = theta + step_size * N(0, I), accept with
    probability alpha = min(1, p(theta') q(theta | theta') / (p(theta) q(theta' | theta))).

    For the symmetric Gaussian proposal q(theta' | theta) = q(theta | theta'),
    so the q terms cancel and log_proposal_ratio is 0. A caller using an
    asymmetric proposal supplies log q(theta | theta') - log q(theta' | theta).
    """
    proposal = theta + step_size * rng.normal(size=theta.shape)
    log_p_proposal = float(log_target(proposal))
    correction = 0.0 if log_proposal_ratio is None else log_proposal_ratio(theta, proposal)
    # If the proposal has zero density, log_p_proposal is -inf and so is log_alpha.
    log_alpha = min(0.0, log_p_proposal - log_p + correction)
    uniform = float(rng.random())
    accepted = np.log(uniform) < log_alpha
    if accepted:
        return StepResult(proposal, log_p_proposal, log_alpha, uniform, True, proposal, log_p_proposal)
    return StepResult(proposal, log_p_proposal, log_alpha, uniform, False, theta, log_p)


def metropolis_hastings(
    log_target: LogDensity,
    theta0: np.ndarray,
    step_size: float | np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    log_proposal_ratio: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> Chain:
    """Run n_steps of Metropolis-Hastings from theta0 and return the whole chain.

    The first sample is the state after the first step, not theta0 itself.
    """
    theta = np.asarray(theta0, dtype=float)
    log_p = float(log_target(theta))
    if not np.isfinite(log_p):
        raise ValueError("theta0 has zero density under the target; start inside the support")

    samples = np.empty((n_steps, theta.size))
    log_probs = np.empty(n_steps)
    accepted = np.empty(n_steps, dtype=bool)
    for i in range(n_steps):
        result = metropolis_step(log_target, theta, log_p, step_size, rng, log_proposal_ratio)
        theta, log_p = result.theta, result.log_p
        samples[i] = theta
        log_probs[i] = log_p
        accepted[i] = result.accepted
    return Chain(samples=samples, log_probs=log_probs, accepted=accepted)