from .TFBM2 import TFBM
from scipy.special import gamma

def mittag_leffler(alpha, beta, gamm, z, tolerance=1e-20):
    prev_sum = 0.0
    k = 1
    factorial_k = 1
    result = (gamma(gamm)) / (factorial_k * gamma(beta))
    while abs(result - prev_sum) > tolerance:
        prev_sum = result
        term = (gamma(gamm + k) * z**k) / (factorial_k * gamma(alpha * k + beta))
        k += 1
        factorial_k *= k
        result += term
    return result / gamma(gamm)

class TFBM3(TFBM):
    def __init__(self, T, N, H, lambd, gamma_H=1, method="davies-harte"):
        super().__init__(T, N, H, lambd, gamma_H, method)
        self.cov_matrices_dir = "cov_matrices_tfbm3"
        self._gamm = 1 - 2*self.H
        self._alpha = 1
        self._beta = 3 - 2*self.H
        self._exponent = 2-2*self.H
        self.cov_matrices_dir = "cov_matrices_tfbm3"

    def ct_2(self, t):
        return 2 *  t**(self._exponent) * mittag_leffler(self._alpha,
                                                         self._beta,
                                                         self._gamm, -t/self.lambd)

    