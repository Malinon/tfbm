import numpy as np
from scipy.special import gamma
from mpmath import hyper
import os

class TFBMII:
    def __init__(self, T, N, H, lambd, method="davies-harte"):
        self.H = H
        self.lambd = lambd
        self.ts = np.linspace(0, T, N+1)
        self.N = N + 1
        self.T = T
        self.method = method
        self.n = N
        self.dt = self.ts[2] - self.ts[1]

    def ct_2(self, t):
        Ct = ((-2*gamma(self.H)*(self.lambd)**(-2*self.H))/(np.sqrt(np.pi)*gamma(self.H-0.5)))*(1-hyper([1, -0.5], [1-self.H, 0.5, 1], ((self.lambd**2) * (t**2))/4)) \
             + (np.abs(t)**(2*self.H) * ((gamma(1-self.H))/(np.sqrt(np.pi)*self.H*(2**(2*self.H)) * gamma(self.H+0.5))) * hyper([1, self.H-0.5], [1, self.H+1, self.H+0.5], ((self.lambd**2) * (t**2))/4))
        return Ct
    
    def covariance(self, t, s):
        return 0.5*(self.ct_2(t) + self.ct_2(s) - self.ct_2(t-s))

    def increment_autocovariance(self, k, dt=1):
        return 0.5 * (self.ct_2(k+dt) - (2 * self.ct_2(k)) + self.ct_2(k-dt))

    def covariance_matrix(self):
        sigma = np.zeros((len(self.ts), len(self.ts)))
        for t in range(len(self.ts)):
            for s in range(len(self.ts)):
                sigma[t][s] = self.covariance(self.ts[t], self.ts[s])
        return sigma
    
    def generate_and_save_cov(self):
        sigma = self.covariance_matrix()
        h_str = str(self.H).replace('.', '_')
        l_str = str(self.lambd).replace('.', '_')
        np.savetxt(f"cov_matrices_tfbm2/H_{h_str}_l_{l_str}_T_{str(self.T)}_N_{str(self.N)}.txt", sigma)
        return sigma

    def load_cov_matrix(self):
        h_str = str(self.H).replace('.', '_')
        l_str = str(self.lambd).replace('.', '_')
        filename = f"cov_matrices_tfbm2/H_{h_str}_l_{l_str}_T_{str(self.T)}_N_{str(self.N)}.txt"
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
        Z = np.random.normal(0, 1, 2*self.n)
        Y = np.zeros(2*self.n, dtype=complex)
        Y[0] = np.sqrt(eigenvals[0]) * Z[0]
        Y[1:int(self.n)] = np.array([np.sqrt(eigenvals[k]/2) for k in range(1, self.n)]) * (np.array([Z[k] for k in range(1, 2*self.n-1, 2)]) + 1j * np.array([Z[k] for k in range(2, 2*self.n, 2)]))
        Y[int(self.n)] = np.sqrt(eigenvals[int(self.n)])*Z[-1]
        Y[int(self.n)+1:] = np.conj([Y[k] for k in range(int(self.n)-1, 0, -1)])

        X = np.real(1/np.sqrt(self.n*2) * np.fft.fft(Y))[:self.n]
        return X

    def generate(self, get_increments=False):
        if self.method == "davies-harte":
            self.t1_ = list(self.ts)
            self.t2_ = list(reversed(self.ts[1:-1]))
            gammas = [self.increment_autocovariance(t, self.ts[2]-self.ts[1]) for t in self.t1_+self.t2_]
            s = np.fft.fft(gammas).real
            if not all(si > 0 for si in s):
                raise Exception("Sorry, no numbers below zero are allowed")
            else:
                incr = self.dh(s)
                if get_increments:
                    return np.insert(np.cumsum(incr), 0, [0], axis=None), incr
                else:
                    return np.insert(np.cumsum(incr), 0, [0], axis=None)
        else:
            ts = self.ts[1:]
            Z = np.random.normal(size=len(ts))
            L = self.cholesky_decomp_from_file()
            sample = L@Z
            sample = np.insert(sample, 0, [0], axis=None)
            if get_increments:
                increments = [sample[j+1] - sample[j] for j in range(len(sample)-1)]
                return sample, increments
            return sample
    
    def generate_samples(self, num_of_samples, get_increments=False):
        if self.method == "davies-harte":
            self.t1_ = list(self.ts)
            self.t2_ = list(reversed(self.ts[1:-1]))
            gammas = [self.increment_autocovariance(t, self.ts[2]-self.ts[1]) for t in self.t1_+self.t2_]
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
            Z = [np.random.normal(size=len(ts)) for _ in range(num_of_samples)]
            L = self.cholesky_decomp_from_file()
            samples = [np.insert(L@Z[i], 0, [0], axis=None) for i in range(num_of_samples)]
            increments = list(np.zeros(len(samples)))
            for i, sample in enumerate(samples):
                increments[i] = [sample[j+1] - sample[j] for j in range(len(sample)-1)]
        if get_increments:
            return samples, increments
        else:
            return samples
