import numpy as np
import os


class TFBM:
    """ Abstract class representing generator of TFBM process """
    def __init__(self, T, N, H, lambd,, method="davies-harte"):
        self.H = H
        self.lambd = lambd
        self.ts = np.linspace(0, T, N+1)
        self.N = N + 1
        self.T = T
        self.method = method
        self.n = N
        self.dt = self.ts[2] - self.ts[1]

    def covariance_matrix(self):
        sigma = np.zeros((len(self.ts), len(self.ts)))
        ct_2_values = [float(self.ct_2(t)) for t in self.ts]
        for t in range(len(self.ts)):
            for s in range(len(self.ts)):
                sigma[t,s] = (ct_2_values[t] + ct_2_values[s] - ct_2_values[abs(t-s)]) / 2
        return sigma
    
    def generate_and_save_cov(self):
        sigma = self.covariance_matrix()
        h_str = str(self.H).replace('.', '_')
        l_str = str(self.lambd).replace('.', '_')
        np.savetxt(self.cov_matrices_dir + f"/H_{h_str}_l_{l_str}_T_{str(self.T)}_N_{str(self.N)}.txt", sigma)
        return sigma

    def load_cov_matrix(self):
        h_str = str(self.H).replace('.', '_')
        l_str = str(self.lambd).replace('.', '_')
        filename = self.cov_matrices_dir + f"/H_{h_str}_l_{l_str}_T_{str(self.T)}_N_{str(self.N)}.txt"
        if os.path.isfile(filename):
            sigma = np.loadtxt(filename)
        else:
            sigma = self.generate_and_save_cov()
        return sigma
    
    def cholesky_decomp_from_file(self):
        sigma = self.load_cov_matrix()
        sigma = np.delete(sigma, 0, 0)
        sigma = np.delete(sigma, 0, 1)
        return np.linalg.cholesky(sigma)

    def dh(self, eigenvals):
        Z_even = np.random.normal(0, 1, self.n - 1)
        Z_odd = np.random.normal(0, 1, self.n - 1)
        Y = np.zeros(2*self.n, dtype=complex)
        Y[1:int(self.n)] = Z_odd + 1j * Z_even
        Y[int(self.n)+1:] = np.conj(Y[1:int(self.n)][::-1])
        Y[0] = np.sqrt(2) * np.random.normal(0, 1)
        Y[int(self.n)] = np.sqrt(2) * np.random.normal(0, 1)
        Y = np.sqrt(eigenvals) * Y

        X = np.real(1/np.sqrt(self.n*2) * np.fft.fft(Y))[:self.n]
        return X
    
    def generate_samples(self, num_of_samples, get_increments=False):
        if self.method == "davies-harte":
            self.t1_ = list(self.ts)
            self.t2_ = list(reversed(self.ts[1:-1]))
            ct_2_values = [self.ct_2(t) for t in self.t1_ + [self.t1_[-1] + self.dt]]
            gammas = np.zeros(2 * (self.N -1))#[3] * (2 * (self.N - 1))
            for k in range(1, self.N-1):
                gammas[k] = 0.5 * (ct_2_values[k + 1] - (2 * ct_2_values[k]) + ct_2_values[abs(k-1)])
                gammas[-k] = gammas[k]
            gammas[self.N -1] = 0.5 * (ct_2_values[self.N] - (2 * ct_2_values[self.N-1]) + ct_2_values[abs(self.N-2)])
            gammas[0] = 0.5 * (ct_2_values[1] - (2 * ct_2_values[0]) + ct_2_values[1])
            gammas[-1] = gammas[1]
            s = np.fft.fft(gammas).real
            if not all(si > 0 for si in s):
                print(f"H: {self.H}, lambda: {self.lambd}")
                print("Switching to cholesky method")
                self.method = "cholesky"
                samples = self.generate_samples(num_of_samples, get_increments)
            else:
                samples = list(np.zeros(num_of_samples))
                increments = list(np.zeros(num_of_samples))
                for i in range(num_of_samples):
                    incr = self.dh(s)
                    samples[i] = np.insert(np.cumsum(incr), 0, [0], axis=None)
                    increments[i] = incr
        else:
            ts = self.ts[1:]
            Z = np.random.normal(size=(len(ts), num_of_samples))
            L = self.cholesky_decomp_from_file()
            samples = np.insert(np.matmul(L,Z), 0, [0], axis=0)
            increments = np.diff(samples, axis=0)
        if get_increments:
            return samples, increments
        else:
            return samples
