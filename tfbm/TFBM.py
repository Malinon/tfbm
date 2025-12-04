import numpy as np
import os


class TFBM:
    """ Abstract class representing generator of TFBM process """
    def __init__(self, T, N, H, lambd, method="davies-harte", save_cov_matrix=True):
        """"
        Parameters:
        T: float
            Time horizon
        N: int
            Number of time steps
        H: float
            Hurst exponent
        lambd: float
            Tempering parameter
        method: str
            Method of generating TFBM process
        save_cov_matrix: bool
            Whether to save covariance matrix to file or not (in Cholesky method)
        """
        self._validate_parameters(T, N, H, lambd, method)
        ## Hurst exponent
        self.H = H
        ## Tempering parameter
        self.lambd = lambd
        ## Times at which process is sampled
        self.ts = np.linspace(0, T, N+1) 
        ## Time horizon
        self.T = T
        ## Name of method of generating TFBM process
        self.method = method
        ## Whether to save covariance matrix to file or not (in Cholesky method)
        self.save_cov_matrix = save_cov_matrix
        ## Number of time steps
        self.n = N
        ## Time step size
        self.dt = self.ts[2] - self.ts[1]
        
    
    def _validate_parameters(self, T, N, H, lambd, method):
        """ Validates parameters of TFBM process """
        if H <= 0:
            raise ValueError("Hurst exponent must be positive")
        if  lambd <= 0:
            raise ValueError("Tempering parameter must be positive")
        if T <= 0:
            raise ValueError("Time horizon must be positive")
        if N <= 0:
            raise ValueError("Number of time steps must be positive integer")
        allowed_methods = ["davies-harte", "cholesky"]
        if method not in allowed_methods:
            raise ValueError(f"Method must be one of {allowed_methods}")

    def covariance_matrix(self):
        """ Generates covariance matrix of TFBM process """
        sigma = np.zeros((len(self.ts), len(self.ts))) # Covariance matrix buffer
        ct_2_values = [float(self.ct_2(t)) for t in self.ts]
        for t in range(len(self.ts)):
            for s in range(len(self.ts)):
                sigma[t,s] = (ct_2_values[t] + ct_2_values[s] - ct_2_values[abs(t-s)]) / 2
        return sigma
    
    def _generate_covariance_filename(self):
        """ Generates filename for covariance matrix of TFBM process """
        h_str = str(self.H).replace('.', '_')
        l_str = str(self.lambd).replace('.', '_')
        return f"H_{h_str}_l_{l_str}_T_{str(self.T)}_N_{str(self.n)}.txt"

    def _load_cov_matrix(self):
        """ Loads covariance matrix of TFBM process from file """
        filename = os.path.join(self.self.cov_matrices_dir, self._generate_covariance_filename())
        if os.path.isfile(filename):
            sigma = np.loadtxt(filename)
        else:
            sigma = self.covariance_matrix()
            if self.save_cov_matrix:
                os.makedeirs(self.self.cov_matrices_dir, exist_ok=True)
                np.savetxt(filename, sigma)
        return sigma
    
    def _get_cholesky_decomposition(self):
        """ Returns Cholesky decomposition of covariance matrix of TFBM process
            If matix is already saved to file, it loads it from there """
        sigma = self._load_cov_matrix()
        # It looks ok, but discuss with advisor about this fragment
        sigma = np.delete(sigma, 0, 0)
        sigma = np.delete(sigma, 0, 1)
        return np.linalg.cholesky(sigma)

    def _generate_dh_increments(self, eigenvals):
        Z_even = np.random.normal(0, 1, self.n - 1)
        Z_odd = np.random.normal(0, 1, self.n - 1)

        Y = np.zeros(2*self.n, dtype=complex)
        Y[1:self.n] = (Z_odd + 1j * Z_even) / np.sqrt(2)
        Y[self.n + 1:] = np.conj(Y[1:self.n][::-1])
        Y[0] = np.random.normal(0, 1)
        Y[self.n] =  np.random.normal(0, 1)
        
        Y = np.sqrt(eigenvals) * Y

        X = np.real(np.fft.fft(Y) / np.sqrt(self.n * 2))[:self.n]
        return X
    
    def _generate_row_of_cirulant_matrix(self):
        """ Generates row of circulant matrix which is used in Davies-Harte method """
        # Cache Ct_2 values
        ct_2_values = [self.ct_2(t) for t in self.ts] + [self.ct_2(self.ts[-1] + self.dt)]
        autocovariance = lambda k: 0.5 * (ct_2_values[k + 1] - (2 * ct_2_values[k]) + ct_2_values[abs(k - 1)])

        row = np.zeros(2 * self.n)
        for k in range(1, self.n):
            row[k] = autocovariance(k)
            row[-k] = row[k]
        row[self.n] = autocovariance(self.n)
        row[0] = autocovariance(0)
    
        return row
    
    def generate_samples(self, num_of_samples, get_increments=False):
        """
        Generates samples of TFBM process
        Parameters:
        num_of_samples: int
            Number of samples to generate
        get_increments: bool
            Whether to return increments of the process along with samples
        Returns:
        samples: np.ndarray
            Generated samples of TFBM process (shape: (num_of_samples, N + 1))
        increments: np.ndarray
            Increments of the process (if get_increments is True)
        """
        if self.method == "davies-harte":
            # Use Davies-Harte method to generate Tempered Fractional Gaussian Noise and transform it to TFBM

            row = self._generate_row_of_cirulant_matrix()
            # Find eigenvalues of circulant matrix
            eigenvals = np.fft.fft(row).real

            if not all(si > 0 for si in eigenvals):
                print(f"H: {self.H}, lambda: {self.lambd}")
                print("Switching to cholesky method")
                self.method = "cholesky"
                samples = self.generate_samples(num_of_samples, get_increments)
            else:
                samples = []
                increments = []
                for i in range(num_of_samples):
                    incr = self._generate_dh_increments(eigenvals)
                    samples.append(np.insert(np.cumsum(incr), 0, [0], axis=None))
                    increments.append(incr)
                samples = np.array(samples)
                increments = np.array(increments)
        else:
            # Generate samples using Cholesky method
            
            Z = np.random.normal(size=(self.n, num_of_samples)) # Generate an (n + 1) column vector of i.i.d. standard normal r.v.
            L = self._get_cholesky_decomposition()
            
            # Perform sum 4.4 from Asmussen (It works, because L is lower triangular matrix)
            # and insert 0 at the beginning of each trajectory
            samples = np.transpose(np.insert(np.matmul(L,Z), 0, [0], axis=0))
            increments = np.diff(samples, axis=1)
        if get_increments:
            return samples, increments
        else:
            return samples
