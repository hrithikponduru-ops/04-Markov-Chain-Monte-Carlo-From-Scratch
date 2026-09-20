"""On a finite state space the Metropolis chain is a matrix, so the theory can
be checked to machine precision rather than to Monte Carlo error.

Detailed balance pi_i P_ij = pi_j P_ji implies stationarity pi P = pi.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcmc.discrete import (  # noqa: E402
    detailed_balance_residual,
    metropolis_transition_matrix,
    neighbour_proposal,
    simulate_discrete_chain,
    stationarity_residual,
)

@pytest.fixture(params=[np.array([0.2, 0.5, 0.3]), None])
def target(request):
    if request.param is not None:
        return request.param
    rng = np.random.default_rng(5)
    return rng.random(6) + 0.1  # six states, unnormalised, all positive

def test_rows_of_transition_matrix_sum_to_one(target):
    P = metropolis_transition_matrix(target, neighbour_proposal(len(target)))
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-14)
    assert np.all(P >= 0)

def test_detailed_balance_holds_exactly(target):
    P = metropolis_transition_matrix(target, neighbour_proposal(len(target)))
    pi = target / target.sum()
    assert detailed_balance_residual(pi, P) < 1e-12

def test_target_is_stationary(target):
    P = metropolis_transition_matrix(target, neighbour_proposal(len(target)))
    pi = target / target.sum()
    assert stationarity_residual(pi, P) < 1e-12

def test_three_state_matrix_matches_hand_calculation():
    P = metropolis_transition_matrix(np.array([0.2, 0.5, 0.3]), neighbour_proposal(3))
    expected = np.array([[0.5, 0.5, 0.0], [0.2, 0.5, 0.3], [0.0, 0.5, 0.5]])
    assert np.allclose(P, expected, atol=1e-15)

def test_simulated_frequencies_approach_target(target):
    P = metropolis_transition_matrix(target, neighbour_proposal(len(target)))
    pi = target / target.sum()
    states = simulate_discrete_chain(P, 0, 200_000, np.random.default_rng(0))
    freq = np.bincount(states, minlength=len(pi)) / len(states)
    assert np.allclose(freq, pi, atol=0.01)