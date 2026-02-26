from .TFBM import TFBM
import numpy as np
from scipy.special import gamma, kv

class TFBM1(TFBM):
    """ Class representing generator of TFBM I process (see doi:10.1016/j.spl.2013.06.016) """
    def __init__(self, T:float, N:int, H:float, lambd:float, method:str="davies-harte",  save_cov_matrix:bool=True, allow_approximation:bool=False, max_embed_exponent:int=1):
        super().__init__(T, N, H, lambd, method, save_cov_matrix, allow_approximation, max_embed_exponent)
        self.cov_matrices_dir = "cov_matrices_tfbm1"
        self._ct_2_additive_part = ((2*gamma(2*self.H))/(2*self.lambd)**(2*self.H))
        self._ct_2_multiplicative_part = ((2*gamma(self.H+0.5)) / np.sqrt(np.pi)) / ((2*self.lambd)**(self.H))


    def _ct_2(self, t:float) -> float:
        ##### condition from https://www.researchgate.net/publication/259744062_Tempered_fractional_Brownian_motion
        if t == 0:
            Ct = 0
        else:
            Ct = (self._ct_2_additive_part - (abs(t)**self.H)
                * self._ct_2_multiplicative_part * kv(self.H, abs(self.lambd*t)))
        return Ct

