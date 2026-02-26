from .TFBM import TFBM

from .mittag_leffer import mittag_leffer

class TFBM3(TFBM):
    """ Class representing generator of TFBM III process (see doi:10.1088/1367-2630/aae4b2) """
    def __init__(self, T:float, N:int, H:float, lambd:float, method:str="davies-harte"):
        super().__init__(T, N, H, lambd, method)
        self.cov_matrices_dir = "cov_matrices_tfbm3"
        self._exponent = 2 - 2 * self.H
    
    def _validate_parameters(self, T:float, N:int, H:float, lambd:float, method:str):
        """ Validates parameters of TFBM3 process """
        super()._validate_parameters(T, N, H, lambd, method)
        if H >= 1 or H < 0.5:
            raise ValueError("Hurst exponent must be in [0.5, 1) for TFBM III")

    def _ct_2(self, t:float) -> float:
        # Assumption k_B  * T / (m * gamma_H) = 1
        return 2 *  t**(self._exponent) * float(mittag_leffer(self.H, -t / self.lambd))
    