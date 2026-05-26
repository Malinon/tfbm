import pywt

import numpy as np
import scipy.special as sp

def wavelet_variance(trajectory: np.ndarray, wavelet: str ='haar', levels: int =4) -> np.ndarray:
    """
    Calculate the wavelet variance of a trajectory.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory data as array.
    wavelet : str
        The type of wavelet to use.
    levels : int
        The number of decomposition levels.

    Returns
    -------
    np.ndarray
        The wavelet variance for each level.
    """

    detailed_coffs = pywt.wavedec(trajectory[:,0], wavelet, level=levels)[1:]  # Skip approximation coefficients
    wavelet_vars = []
    for i in range(levels):
        detailed_coeff = detailed_coffs[-(i+1)]
        wavelet_vars.append(np.sum(detailed_coeff * detailed_coeff / len(detailed_coeff)))

    return np.array(wavelet_vars)


def bias_corrected_log_spectrum(H, sigma, level, n, l  = 10):
    C_H = np.sqrt(np.pi / (H * np.gamma(2 * H) * np.sin(np.pi * H)))
    n_j = n / 2 ** level


def wavelet_estimator(trajectory: np.ndarray, wavelet: str ='haar', levels: int = 4):
    wavelet_variance = wavelet_variance(trajectory, wavelet, levels)
    target_function = lambda H, sigma: 