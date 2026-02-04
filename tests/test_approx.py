from tfbm.approx import find_optimal_eigenvals, wood_chan
import pytest
import numpy as np

def cov_minus_one(_):
    return -1

def cov_one(_):
    return 1

def test_fail_when_approximation_not_allowed():
    pytest.raises(ValueError, find_optimal_eigenvals, cov_minus_one, 10, 100, approximate=False)

def test_return_zeros_for_negative_covariance_with_approximation():
    max_embedding_exp = 5
    embdedding_exp, eigenvals = find_optimal_eigenvals(cov_minus_one, max_embedding_exp, 100, approximate=True)
    assert (eigenvals == 0).all()
    assert embdedding_exp == max_embedding_exp

def test_constant_covariance_result_in_const_increments():
    max_embedding_exp = 3
    traj_length = 100
    _, increments = wood_chan(cov_one, max_embedding_exp, traj_length, approximate=False)
    expected_inc = np.array([increments[0]] * len(increments))
    assert increments.shape == (traj_length,)
    assert np.allclose(increments, expected_inc)