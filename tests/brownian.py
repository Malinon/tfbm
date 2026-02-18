from tfbm import TFBM


class BrownianMotion(TFBM):
    def __init__(self, T, N, H, lambd, method="davies-harte"):
        super().__init__(T, N, H, lambd, method)
        self.cov_matrices_dir = "cov_matrices_brownian"

    def ct_2(self, t):
        return t