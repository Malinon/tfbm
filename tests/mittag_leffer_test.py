import pytest
import scipy.special as sc
import numpy as np
import sys
sys.path.append("../tfbm")  # Adjust path to import from parent directory
from mittag_leffer import gamma_inc
from mittag_leffer import mittag_leffer as ml

PRECISION = 1e-10


def test_gamma_inc_like_scipy():
    # Test gamma_inc against scipy.special.gamma
    for a in [0.1, 0.5, 1.0, 2.0, 3.0]:
        for z in [0.1, 1.0, 10.0]:
            assert abs(gamma_inc(a, z) - sc.gammainc(a, z) * sc.gamma(a)) < PRECISION

def test_gamma_inc_one():
    args = [0.1, 0.5, 1.0, 2.0, 3.0]
    vals = 1 - np.exp(-np.array(args))
    for z, val in zip(args, vals):
        assert abs(gamma_inc(1, z) - val) < PRECISION


def simplified_mittag_leffler(H, z, tolerance=1e-20):
    prev_sum = 0.0
    k = 1
    gamm = 1 - 2 * H
    beta = 3 - 2 * H
    if H == 0.5:
        # Extreme case H = 0.5
        return 1.0 / sc.gamma(gamm)
    result = (sc.gamma(gamm)) / (sc.gamma(beta))
    while abs(result - prev_sum) > tolerance:
        prev_sum = result
        term = ((z**k) / (sc.factorial(k))) / ((k + 2 - 2 * H) * (k + gamm)) 
        k += 1
        if term != np.inf and term in [np.nan] and term != -np.inf:
            result += term
        else:
            break
    return result / sc.gamma(gamm)

def test_mittag_leffler_for_small_arg():
    # For small z, summation converges quickly
    zs = [-0.01, -0.005, -0.001]
    Hs = np.linspace(0.55, 0.95, 10)
    for H in Hs:
        for z in zs:
            print("Mittag-Leffler for H={}, z={}".format(H, z))
            print("My: ", ml(H, z))
            assert abs(ml(H, z) - simplified_mittag_leffler(H, z)) / abs(simplified_mittag_leffler(H, z)) < 0.01


