import numpy as np
from scipy.special import gamma
from mpmath import hyper

from .TFBM import TFBM

class TFBM2(TFBM):

    def __init__(self, T, N, H, lambd, gamma_H=1, method="davies-harte"):
        super().__init__(T, N, H, lambd, gamma_H, method)
        self.cov_matrices_dir = "cov_matrices_tfbm2"
        self._ct2_first_multiplicative =  ((-2*gamma(self.H)*(self.lambd)**(-2*self.H))/(np.sqrt(np.pi)*gamma(self.H-0.5)))
        self._ct2_second_multiplicative = ((gamma(1-self.H))/(np.sqrt(np.pi)*self.H*(2**(2*self.H)) * gamma(self.H+0.5)))
        self._lambd_squared = self.lambd ** 2

    def ct_2(self, t):
        return (self._ct2_first_multiplicative
             * (1-hyper([1, -0.5], [1-self.H, 0.5, 1], (self._lambd_squared * (t**2))/4))
             + (np.abs(t)**(2*self.H) * self._ct2_second_multiplicative
             * hyper([1, self.H-0.5], [1, self.H+1, self.H+0.5], (self._lambd_squared * (t**2))/4)))
