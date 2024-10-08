from TFBM import TFBM
import numpy as np
from scipy.special import gamma, kv

class TFBM1(TFBM):

    def __init__(self, T, N, H, lambd, gamma_H=1, method="davies-harte"):
        super().__init__(T, N, H, lambd, gamma_H, method)
        self.cov_matrices_dir = "cov_matrices_tfbm3"
        self._ct_2_additive_part = ((2*gamma(2*self.H))/(2*self.lambd)**(2*self.H))
        self._ct_2_multiplicative_part = ((2*gamma(self.H+0.5))/np.sqrt(np.pi)) * (1/((2*self.lambd)**(self.H)))


    def ct_2(self, t):
        ##### condition from https://www.researchgate.net/publication/259744062_Tempered_fractional_Brownian_motion
        if t == 0:
            Ct = 0
        else:
            Ct = (self._ct_2_additive_part - (abs(t)**self.H)
                * self._ct_2_multiplicative_part * kv(self.H, abs(self.lambd*t)))
        return Ct


if __name__ == "__main__":
    T = 10
    N = 100
    H = 0.7
    lambd = 0.3
    generator = TFBM1(T, N, H, lambd)
    num_of_samples = 1000
    generator.generate_samples(num_of_samples)
    import cProfile
    cProfile.run('generator.generate_samples(num_of_samples)', 'profile_results_tfbm1_new')
    import pstats
    stats = pstats.Stats('profile_results_tfbm1_new')
    stats.sort_stats('time').print_stats()