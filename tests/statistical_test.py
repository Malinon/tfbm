import pytest
import numpy as np

from scipy.stats import kstest
from scipy.stats import norm

from tfbm import TFBM1, TFBM2, TFBM3
from tfbm.tests import avcf_test, dma_test, tamsd_test

from brownian import BrownianMotion


TFBMS = [TFBM1, TFBM2, TFBM3]
STAT_TEST_FUNCTIONS = [avcf_test, dma_test, tamsd_test]
H = 0.7
T = 10
LAMBDA = 0.5

MONTE_CARLO_STEPS = 100

def class_to_type(x):
    if x == TFBM1:
        return "1"
    elif x == TFBM2:
        return "2"
    elif x == TFBM3:
        return "3"
    else:
        raise ValueError("Unknown class")

@pytest.mark.parametrize("tfbm_class", TFBMS)
@pytest.mark.parametrize("stat_test_function", STAT_TEST_FUNCTIONS)
def test_tfbm_pass_quadratic_test(tfbm_class, stat_test_function):
    np.random.seed(1905)
    trajectories = tfbm_class(T, N=100, H=H, lambd=LAMBDA).generate_samples(MONTE_CARLO_STEPS)
    p_vals= [stat_test_function(traj, class_to_type(tfbm_class), 0.75, LAMBDA, lag=5, T=T, monte_carlo_steps=MONTE_CARLO_STEPS) for traj in trajectories]
    median_p_val = np.median(p_vals)
    assert median_p_val > 0.1, f"Median p-value {median_p_val} is too low for {tfbm_class.__name__} with {stat_test_function.__name__}"

def test_brownian_increments_are_ok():
    np.random.seed(3721)
    trajs = BrownianMotion(T, N=100, H=H, lambd=LAMBDA).generate_samples(MONTE_CARLO_STEPS)
    expected_cdf = norm(loc=0, scale=np.sqrt(T / 100)).cdf
    p_vals = [kstest(np.diff(traj), expected_cdf).pvalue for traj in trajs]
    median_p_val = np.median(p_vals)
    assert median_p_val > 0.1, f"Median p-value {median_p_val} is too low for BrownianMotion increments"
    