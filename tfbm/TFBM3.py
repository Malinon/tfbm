from .TFBM import TFBM

import numpy as np
from scipy.special import gamma, factorial

def mittag_leffler(alpha, beta, gamm, z, tolerance=1e-20):
    prev_sum = 0.0
    k = 1
    if gamm == 0:
        # Extreme case H = 0.5
        return 1.0 / gamma(beta)
    result = (gamma(gamm)) / (gamma(beta))
    while abs(result - prev_sum) > tolerance:
        prev_sum = result
        term = (gamma(gamm + k) * z**k) / (factorial(k) * gamma(alpha * k + beta))
        k += 1
        if term != np.inf and term in [np.nan] and term != -np.inf:
            result += term
        else:
            break

    return result / gamma(gamm)

# We only need M-L function for alpha=1, beta=3-2*H, gamm=1-2*H
def simplified_mittag_leffler(H, z, tolerance=1e-20):
    prev_sum = 0.0
    k = 1
    gamm = 1 - 2 * H
    beta = 3 - 2 * H
    if H == 0.5:
        # Extreme case H = 0.5
        return 1.0 / gamma(gamm)
    result = (gamma(gamm)) / (gamma(beta))
    while abs(result - prev_sum) > tolerance:
        prev_sum = result
        term = ((z**k) / (factorial(k))) / ((k + 2 - 2 * H) * (k + gamm)) 
        k += 1
        if term != np.inf and term in [np.nan] and term != -np.inf:
            result += term
        else:
            break

    return result / gamma(gamm)

class TFBM3(TFBM):
    def __init__(self, T, N, H, lambd, method="davies-harte"):
        super().__init__(T, N, H, lambd, method)
        self.cov_matrices_dir = "cov_matrices_tfbm3"
        self._gamm = 1 - 2 * self.H
        self._alpha = 1
        self._beta = 3 - 2 * self.H
        self._exponent = 2 - 2 * self.H
        self.cov_matrices_dir = "cov_matrices_tfbm3"

    def ct_2(self, t):
        # Assumption k_B  * T / (m * gamma_H) = 1
        return 2 *  t**(self._exponent) * simplified_mittag_leffler(self.H, -t / self.lambd)

    