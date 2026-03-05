from tfbm.approx import find_optimal_eigenvals, wood_chan
from tfbm import TFBM1, TFBM2, TFBM3
import pytest
import numpy as np


TFBMS = [TFBM1, TFBM2, TFBM3]

def cov_minus_one(_):
    return -1

def cov_one(_):
    return 1

def test_fail_when_approximation_not_allowed():
    pytest.raises(ValueError, find_optimal_eigenvals, cov_minus_one, 10, 100, approximate=False)

@pytest.mark.parametrize("tfbm_class", TFBMS)
def test_wood_chan_works(tfbm_class):
    traj_length = 100
    gen=tfbm_class(1, traj_length, 0.75, 0.1)
    covariance_function = lambda t: 0.5 * (gen.ct_2(t * gen.T + gen.dt) - 2 * gen.ct_2(t * gen.T) + gen.ct_2(abs(t *gen.T - gen.dt)))
    trajectory, increments = wood_chan(covariance_function, 7, traj_length, approximate=False)
    assert trajectory.shape == (traj_length,)
    assert increments.shape == (traj_length,)
    assert np.allclose(np.cumsum(increments), trajectory)

@pytest.mark.parametrize("tfbm_class", TFBMS)
def test_looking_for_optimal_eigenvals_works(tfbm_class):
    counter = 0
    traj_length = 100
    gen=tfbm_class(1, traj_length, 0.75, 0.1)
    def cov_func(t):
        nonlocal counter
        if counter >= traj_length:
            return 0.5 * (gen.ct_2(t * gen.T + gen.dt) - 2 * gen.ct_2(t * gen.T) + gen.ct_2(abs(t *gen.T - gen.dt)))
        else:
            counter += 1
            return -1
    embed_exp, optimal_eigenvals = find_optimal_eigenvals(cov_func, max_embedding_exp=10, sample_size=100, approximate=False)
    assert embed_exp == 8
    assert np.all(optimal_eigenvals > 0)


def test_constant_covariance_result_in_const_increments():
    max_embedding_exp = 10
    traj_length = 100
    _, increments = wood_chan(cov_one, max_embedding_exp, traj_length, approximate=False)
    expected_inc = np.array([increments[0]] * len(increments))
    assert increments.shape == (traj_length,)
    assert np.allclose(increments, expected_inc)