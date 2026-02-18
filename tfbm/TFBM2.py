import numpy as np
from scipy.special import gamma
import scipy.integrate as si
from mpmath import hyper
import mpmath as mp

from .TFBM import TFBM

_LIMIT_FOR_FORMULA = 25
_EPSILON = 1e-10

class TFBM2(TFBM):
    """ Class representing generator of TFBM II process (doi: 10.1016/j.spl.2017.08.015) """
    def __init__(self, T, N, H, lambd, method="davies-harte", strategy="flexible"):
        super().__init__(T, N, H, lambd, method)
        self.cov_matrices_dir = "cov_matrices_tfbm2"    
        self._ct2_first_multiplicative =  (-2*gamma(self.H)*((self.lambd)**(-2*self.H))) / (np.sqrt(np.pi)*gamma(self.H-0.5))
        self._ct2_second_multiplicative = (gamma(1-self.H)) / (np.sqrt(np.pi) * self.H * (2**(2*self.H)) * gamma(self.H+0.5))
        self._lambd_squared = self.lambd ** 2

        # Determine inegration strategy
        if strategy == "flexible":
            self.strict = self._is_integral_problematic()
        elif strategy == "strict":
            self.strict = True
        elif strategy == "fast":
            self.strict = False
        else:
            raise ValueError("Invalid strategy. Choose from 'strict', 'fast', or 'flexible'.") 
    
    def _integrand(H, lambd, t, omega):
        if np.abs(omega)< _EPSILON:
            return (t**2) * np.pow((lambd**2) + (omega **2), 0.5 - H) / np.pi

        return (( 2 * mp.sin(omega * t / 2) / omega) ** 2)  * np.pow((lambd**2) + (omega **2), 0.5 - H) / np.pi

    def _use_integration(self, t):
        if np.abs(self.H - 0.5) < _EPSILON or np.ceil(self.H - self.H) < _EPSILON:
            # H = 0.5 and integer H are poles of terms in formula, so we need to use integration to compute ct_2
            return True
        if t * self.lambd > _LIMIT_FOR_FORMULA:
            # For large t * lambda, the formula involves hypergeometric functions with large arguments,
            # which can lead to numerical instability. In such cases, we use integration to compute ct_2.
            return True
        return False

    
    def  _is_integral_problematic(self):
        if self.H  < 0.5:
            return True
        # For big H integral is easy to compute because of fast decay of integrand
        return False

    def ct_2(self, t):
        if self._use_integration(t):
            # Using standard formula is not numerically stable, so we compute ct_2 by integrating
            final_integrand = lambda omega: TFBM2._integrand(self.H, self.lambd, t, omega)
            if self.strict:
                # Use mpmath for high-precision integration, which is more accurate but slower
                integral = mp.quadsubdiv(final_integrand, [0, np.inf])
            else:
                # Use scipy integration for faster computation, which may be less accurate for some parameters (small H)
                integral = si.quad(final_integrand, 0, np.inf)[0]
            return float(integral) / np.pi
        else:
            # Used standard formula (Eq. 2.19, https://doi.org/10.1016/j.spl.2017.08.015) with hypergeometric functions
            return float(self._ct2_first_multiplicative
                * (1 - hyper([1, -0.5], [1-self.H, 0.5, 1], (self._lambd_squared * (t**2))/4))
                + ((np.abs(t)**(2*self.H)) * self._ct2_second_multiplicative
                * hyper([1, self.H-0.5], [1, self.H+1, self.H+0.5], (self._lambd_squared * (t**2))/4)))
