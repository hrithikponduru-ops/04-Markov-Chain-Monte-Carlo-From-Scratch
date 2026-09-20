"""Two examples small enough to check with a pencil. Mirrors README.md section 2.

Part 1: Metropolis on three states, where the whole chain is a 3 x 3 matrix.
Part 2: three explicit Metropolis steps on the coin problem.

Run from this folder:  python worked_example.py
"""

import numpy as np

from mcmc.discrete import (
    detailed_balance_residual,
    metropolis_transition_matrix,
    neighbour_proposal,
    stationarity_residual,
)
from mcmc.metropolis import metropolis_step
from mcmc.models import log_beta_binomial_posterior

np.set_printoptions(precision=4, suppress=True)


def part_1_three_states() -> None:
    print("=" * 64)
    print("PART 1: Metropolis on three states, target pi = (0.2, 0.5, 0.3)")
    print("=" * 64)
    
    pi = np.array([0.2, 0.5, 0.3])
    Q = neighbour_proposal(3)
    
    print("\nProposal Q (propose a neighbour with prob 1/2; at the ends, 'stay' takes the other half):")
    print(Q)

    P = metropolis_transition_matrix(pi, Q)
    print("\nAcceptance probabilities alpha_ij = min(1, pi_j / pi_i) for the moves Q allows:")
    print(f"  0 -> 1: min(1, 0.5/0.2) = 1.0       1 -> 0: min(1, 0.2/0.5) = {0.2 / 0.5:.1f}")
    print(f"  1 -> 2: min(1, 0.3/0.5) = {0.3 / 0.5:.1f}       2 -> 1: min(1, 0.5/0.3) = 1.0")
    
    print("\nTransition matrix P_ij = Q_ij * alpha_ij, diagonal = 1 - row sum of the rest:")
    print(P)

    print("\nStationarity: pi P should equal pi")
    print(f"  pi P = {pi @ P}   residual {stationarity_residual(pi, P):.1e}")
    
    print("\nDetailed balance: pi_i P_ij = pi_j P_ji for every pair")
    print(f"  pi_0 P_01 = 0.2 * 0.5 = {pi[0] * P[0, 1]:.2f}    pi_1 P_10 = 0.5 * 0.2 = {pi[1] * P[1, 0]:.2f}")
    print(f"  pi_1 P_12 = 0.5 * 0.3 = {pi[1] * P[1, 2]:.2f}    pi_2 P_21 = 0.3 * 0.5 = {pi[2] * P[2, 1]:.2f}")
    print(f"  residual {detailed_balance_residual(pi, P):.1e}")


def part_2_three_coin_steps() -> None:
    print("\n" + "=" * 64)
    print("PART 2: three Metropolis steps on the coin problem")
    print("  7 heads in 10 flips, prior Beta(2, 2), so log p(theta) = 8 log theta + 4 log(1 - theta)")
    print("=" * 64)
    
    k, n, a, b = 7, 10, 2.0, 2.0
    log_target = lambda t: log_beta_binomial_posterior(t[0], k, n, a, b)  # noqa: E731
    rng = np.random.default_rng(0)
    
    theta = np.array([0.5])
    log_p = log_target(theta)
    
    print(f"\nstart: theta = {theta[0]:.4f}, log p = 8 log 0.5 + 4 log 0.5 = {log_p:.4f}")
    
    for step in range(1, 4):
        result = metropolis_step(log_target, theta, log_p, 0.1, rng)
        print(f"\nstep {step}")
        print(f"  propose theta' = theta + 0.1 * z = {result.proposal[0]:.4f}")
        print(f"  log p(theta')  = {result.log_p_proposal:.4f}")
        print(f"  log alpha      = min(0, {result.log_p_proposal:.4f} - {log_p:.4f}) = {result.log_alpha:.4f}"
              f"\n      -> alpha = {np.exp(result.log_alpha):.4f}")
        print(f"  draw u         = {result.uniform:.4f}")
        
        verdict = "ACCEPT (u < alpha)" if result.accepted else "REJECT (u >= alpha), stay put"
        print(f"  {verdict} -> theta = {result.theta[0]:.4f}")
        
        theta, log_p = result.theta, result.log_p


if __name__ == "__main__":
    part_1_three_states()
    part_2_three_coin_steps()