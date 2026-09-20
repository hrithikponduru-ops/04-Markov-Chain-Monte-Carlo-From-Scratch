"""Metropolis on a finite state space, where the chain is literally a matrix.

With S states the transition probabilities form an S x S matrix P, the
target is a vector pi, and the two theorems behind MCMC become matrix
identities that can be checked to 1e-15:
    detailed balance    pi_i P_ij = pi_j P_ji
    stationarity        pi P = pi
"""

import numpy as np

def neighbour_proposal(n_states: int) -> np.ndarray:
    """Symmetric proposal on a line: move left or right with probability 1/2 each.

    At the two ends, the move that would leave the state space is replaced
    by "stay", which keeps every row summing to one and keeps Q symmetric.
    """
    Q = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in (i - 1, i + 1):
            Q[i, j if 0 <= j < n_states else i] += 0.5
    return Q

def metropolis_transition_matrix(target: np.ndarray, proposal: np.ndarray) -> np.ndarray:
    """P_ij = Q_ij * min(1, pi_j Q_ji / (pi_i Q_ij)) for j != i; P_ii takes the rest.

    The target may be unnormalised: only ratios pi_j / pi_i appear.
    """
    pi = np.asarray(target, dtype=float)
    Q = np.asarray(proposal, dtype=float)
    n = len(pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (pi[None, :] * Q.T) / (pi[:, None] * Q)  # ratio[i, j] = pi_j Q_ji / (pi_i Q_ij)
    alpha = np.where(Q > 0, np.minimum(1.0, ratio), 0.0)
    off_diagonal = Q * alpha
    np.fill_diagonal(off_diagonal, 0.0)
    P = off_diagonal + np.diag(1.0 - off_diagonal.sum(axis=1))
    assert P.shape == (n, n)
    return P

def detailed_balance_residual(pi: np.ndarray, P: np.ndarray) -> float:
    """max over i, j of the absolute value of pi_i P_ij - pi_j P_ji."""
    flow = pi[:, None] * P
    return float(np.abs(flow - flow.T).max())

def stationarity_residual(pi: np.ndarray, P: np.ndarray) -> float:
    """The largest entry of the absolute value of pi P - pi."""
    return float(np.abs(pi @ P - pi).max())

def simulate_discrete_chain(P: np.ndarray, state0: int, n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Run the chain by sampling each next state from the row P[state]."""
    cumulative = np.cumsum(P, axis=1)
    uniforms = rng.random(n_steps)
    states = np.empty(n_steps, dtype=int)
    state = state0
    for t in range(n_steps):
        state = int(np.searchsorted(cumulative[state], uniforms[t]))
        states[t] = min(state, P.shape[0] - 1)  # guard against cumulative[-1] = 1 - 1e-16
        state = states[t]
    return states