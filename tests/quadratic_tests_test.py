import pytest
import numpy as np
from scipy.stats import chi2

from tfbm.tests import avcf_test, dma_test, tamsd_test, common_quadratic_test_subroutine
from tfbm import TFBM1, TFBM2, TFBM3

def _template_const_testing(test_function, test_name):
    trajectory = np.zeros(100)
    for tfbm_type in range(1, 4):
        for lambd in [0.1, 0.5, 1.0]:
            for lag in [1, 5, 10]:
                for T in [1, 5, 10]:
                    p_value = test_function(trajectory, str(tfbm_type), 0.75, lambd, lag, T, monte_carlo_steps=1000)
                    assert p_value < 0.1, f"Constant trajectory test failed for {test_name} with tfbm_type={tfbm_type}, lambd={lambd}, lag={lag}, T={T}"


@pytest.mark.parametrize("tfbm_type", [TFBM1, TFBM2, TFBM3])
@pytest.mark.parametrize("lambd", [0.1, 0.5, 1.0])
@pytest.mark.parametrize("T", [1, 5, 10])
def test_cov_matrix_is_positive_semidefinite_symmetric(tfbm_type, lambd, T):
    gen = tfbm_type(T, N=100, H=0.75, lambd=lambd, method="davies-harte")
    sigma = gen.covariance_matrix()
    eignevals = np.linalg.eigvalsh(sigma)
    assert (eignevals >= -1e-10).all()
    assert np.allclose(sigma, sigma.T)

def test_constant_trajectory_fails_tamsd():
    _template_const_testing(tamsd_test, "TAMSD")

def test_constant_trajectory_fails_dma():
    _template_const_testing(dma_test, "DMA")

def test_constant_trajectory_fails_acvf():
    _template_const_testing(avcf_test, "ACVF")

def test_quantiles_for_chi_squared_are_correct():
    sample_size = 300
    degrees_of_freedom = sample_size
    cov_matrix = np.identity(sample_size)
    stat_matrix = np.identity(sample_size)
    chi_2_quantiles_big = chi2.ppf([0.7, 0.8, 0.9], df=degrees_of_freedom)
    p_values = np.array([common_quadratic_test_subroutine(stat_matrix, q, cov_matrix, 1000) for q in chi_2_quantiles_big])
    pytest.approx((1 -chi_2_quantiles_big) * 2, p_values, abs=1e-1)
    chi_2_quantiles_small = chi2.ppf([0.1, 0.2, 0.3], df=degrees_of_freedom)
    p_values = np.array([common_quadratic_test_subroutine(stat_matrix, q, cov_matrix, 1000) for q in chi_2_quantiles_small])
    pytest.approx(chi_2_quantiles_small / 2, p_values, abs=1e-1)
    
def test_works_well_for_sum_of_squares_of_independent_vars():
    np.random.seed(1848)
    sample_size = 300
    cov_matrix = np.identity(sample_size)
    stat_matrix = np.identity(sample_size)
    test_statistic = np.sum(np.random.normal(0, 1, sample_size)**2)

    p_value = common_quadratic_test_subroutine(stat_matrix, test_statistic, cov_matrix, 1000)
    assert p_value >= 0.1
