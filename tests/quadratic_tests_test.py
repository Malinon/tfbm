import pytest
import numpy as np
from scipy.stats import chi2

from tfbm.tests import avcf_test, dma_test, tamsd_test, quadratic_test

def _template_const_testing(test_function, test_name):
    for tfbm_type in range(1, 4):
        for lambd in [0.1, 0.5, 1.0]:
            for lag in [1, 5, 10]:
                for T in [10, 50, 100]:
                    p_value = test_function(tfbm_type, 0.75, lambd, lag, T, monte_carlo_steps=1000)
                    assert p_value > 0.1, f"Constant trajectory test failed for {test_name} with tfbm_type={tfbm_type}, lambd={lambd}, lag={lag}, T={T}"

def constant_trajectory_should_fail_tamsd():
    _template_const_testing(tamsd_test, "TAMSD Test")

def constant_trajectory_should_fail_dma():
    _template_const_testing(dma_test, "DMA Test")

def constant_trajectory_should_fail_acvf():
    _template_const_testing(avcf_test, "ACVF Test")

def quantiles_for_chi_squared_should_be_correct():
    sample_size = 300
    degrees_of_freedom = sample_size
    cov_matrix = np.identity(sample_size)
    stat_matrix = np.identity(sample_size)
    eigenvalues = np.array([1.0 for _ in range(degrees_of_freedom)])
    chi_2_quantiles_big = chi2.ppf([0.7, 0.8, 0.9], df=degrees_of_freedom)
    p_values = np.array([quadratic_test(stat_matrix, q, cov_matrix) for q in chi_2_quantiles_big])
    pytest.approx((1 -chi_2_quantiles_big) * 2, p_values, abs=1e-1)
    chi_2_quantiles_small = chi2.ppf([0.1, 0.2, 0.3], df=degrees_of_freedom)
    p_values = np.array([quadratic_test(stat_matrix, q, cov_matrix) for q in chi_2_quantiles_small])
    pytest.approx(chi_2_quantiles_small / 2, p_values, abs=1e-1)
    
def test_works_well_for_sum_of_squares_of_independent_vars():
    sample_size = 300
    cov_matrix = np.identity(sample_size)
    stat_matrix = np.identity(sample_size)
    test_statistic = np.sum(np.random.normal(0, 1, sample_size)**2)
    p_value = quadratic_test(stat_matrix, test_statistic, cov_matrix)
    assert p_value <= 0.05
