from .TFBM import TFBM

import numpy as np
import mpmath
import numpy as np
from scipy.special import gamma


def gamma_inc(a, z):
    return mpmath.gammainc(z=a, a=0, b=z)

def mittag_leffer(H, z):
    if z == 0:
        # 1 / gamma(beta)
        return 1.0 / gamma(3 - 2 * H)
    if H == 0.5:
        # Extreme case H = 0.5
        return 1.0
    a1 = 1 - 2 * H
    a2 = 2 - 2 * H
    numerator = -(z * gamma_inc(a1, -z) + gamma_inc(a2, -z))
    denominator = gamma(a1) * (-z)**(a2)
    return numerator / denominator

class TFBM3(TFBM):
    def __init__(self, T, N, H, lambd, method="davies-harte"):
        super().__init__(T, N, H, lambd, method)
        self.cov_matrices_dir = "cov_matrices_tfbm3"
        self._exponent = 2 - 2 * self.H
        self.cov_matrices_dir = "cov_matrices_tfbm3"

    def ct_2(self, t):
        # Assumption k_B  * T / (m * gamma_H) = 1
        return 2 *  t**(self._exponent) * mittag_leffer(self.H, -t / self.lambd)

    