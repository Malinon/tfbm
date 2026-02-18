import pytest
from tfbm import TFBM2
import numpy as np


def test_covariance_matrix_are_similar_for_both_strategies():
    H = 1.2
    T = 20
    lambd = 2.0
    N = 5
    gen1 = TFBM2(T, N, H, lambd, method="davies-harte", strategy="strict")
    covariance_matrix = gen1.covariance_matrix()
    gen2 = TFBM2(T, N, H, lambd, method="davies-harte", strategy="fast")
    covariance_matrix_approx = gen2.covariance_matrix()
    assert covariance_matrix.shape == covariance_matrix_approx.shape
    assert np.allclose(covariance_matrix, covariance_matrix_approx, atol=1e-2)

def test_code_works_for_H_breaking_standard_formula():
    H = 1
    T = 1
    lambd = 0.5
    N = 3
    for H in [0.5, 1, 2, 3, 4]:
        gen = TFBM2(T, N, H, lambd, method="davies-harte", strategy="flexible")
        cov_matrix = gen.covariance_matrix()
        eignevals = np.linalg.eigvalsh(cov_matrix)
        assert (eignevals >= -1e-10).all()


def test_fast_computing_is_used_for_big_H():
    H = 10
    T = 1
    lambd = 0.5
    N = 100
    gen = TFBM2(T, N, H, lambd, method="davies-harte", strategy="flexible")
    assert not gen.strict