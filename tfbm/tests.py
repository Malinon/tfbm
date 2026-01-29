from scipy.stats import chi2
from scipy.linalg import sqrtm
import numpy as np

from .TFBM1 import TFBM1
from .TFBM2 import TFBM2
from .TFBM3 import TFBM3

def _generalized_chi_square_samples(eigenvalues, num_samples):
    """ Generates samples from a generalized chi-square distribution by summing chi-square variables weighted by eigenvalues"""
    n = len(eigenvalues)
    samples = np.zeros(num_samples)
    for i in range(num_samples):
        randoms = chi2.rvs(df=1, size=n)
        samples[i] = np.sum(eigenvalues * randoms)
    return samples

def _compute_covariance_matrix(tfbm_type, H_0, lambda_0, N, T):
    if tfbm_type == "1":
        tfbm = TFBM1(H=H_0, lambd=lambda_0, T=T, n=N-1)
    elif tfbm_type == "2":
        tfbm = TFBM2(H=H_0, lambd=lambda_0, T=T, n=N-1)
    elif tfbm_type == "3":
        tfbm = TFBM3(H=H_0, lambd=lambda_0, T=T, n=N-1)
    else:
        raise ValueError("Invalid TFBM type. Must be '1', '2', or '3'.")
    return tfbm.covariance_matrix()

def _dma_covariance(tfbm_type, H_0, lambda_0, N, T, lag):
    basic_cov = _compute_covariance_matrix(tfbm_type, H_0, lambda_0, N, T)
    dma_covariance = np.zeros((N - lag + 1, N - lag + 1))
    for j in range(N - lag + 1):
        for k in range(j+1):
            dma_covariance[j, k] = ( ((1 - 1/lag)**2) * basic_cov[j + lag - 1, k + lag -1]
            + (1/(lag**2) - 1 / lag) * basic_cov[j + lag - 1, k:(k+lag-1)].sum()
            + (1/(lag**2) - 1 / lag) * basic_cov[k + lag - 1, j:(j+lag-1)].sum()
            + (1/(lag**2)) * basic_cov[j:(j+lag-1), k:(k+lag-1)].sum())
            dma_covariance[k, j] = dma_covariance[j, k]
    return dma_covariance

def _acvf_matrix(N, lag):
    if lag == 0:
        return np.identity(N) / N
    
    stat_matrix = np.zeros((N, N))
    val = 1 / (2 * (N - lag))

    for i in range(N):
        if i + lag < N:
            stat_matrix[i, i + lag] = val
        if i - lag >= 0:
            stat_matrix[i, i - lag] = val
    return stat_matrix
            
def _tamsd_matrix(N, lag):
    stat_matrix = np.zeros((N, N))
    for i in range(lag, N):
        stat_matrix[i, i] += 1
        stat_matrix[i, i-lag] += -1
    for i in range(N - lag):
        stat_matrix[i, i] += 1
        stat_matrix[i, i+lag] -= 1

    stat_matrix = stat_matrix  / (N - lag)
    return stat_matrix

def _get_dma(trajectory, lag):
    trajectory_extended = np.insert(trajectory, 0, 0, axis=0)
    trajectory_cumsum = np.cumsum(trajectory_extended, axis=0)
    moving_avg = (trajectory_cumsum[lag:] - trajectory_cumsum[:-lag]) / lag
    detrended_trajectory = trajectory[(lag - 1):] - moving_avg
    dma = np.mean(detrended_trajectory**2)
    return dma

def _quadratic_form_test(eigenvalues, test_statistic, monte_carlo_steps):
    num_samples = monte_carlo_steps
    gen_samples = _generalized_chi_square_samples(eigenvalues, num_samples)
    less_eq_count = np.sum(gen_samples >= test_statistic) / num_samples
    if less_eq_count < 0.5:
        p_value = 2 * less_eq_count
    else:
        p_value = 2 * (1 - less_eq_count)
    return p_value

def _common_quadratic_test_subroutine(stat_matrix, test_statistic, cov_matrix, monte_carlo_steps):
    sqrt_cov_matrix = sqrtm(cov_matrix)
    eigenvals = np.linalg.eigvalsh(sqrt_cov_matrix @ stat_matrix @ sqrt_cov_matrix)
    p_value = _quadratic_form_test(eigenvals, test_statistic, monte_carlo_steps)
    return p_value

def _general_quadratic_form_test(matrix_gen, trajectory, tfbm_type, H_0, lambda_0, lag, monte_carlo_steps):
    N = len(trajectory)
    stat_matrix = matrix_gen(N, lag=lag)
    test_statistic = trajectory.T @ stat_matrix @ trajectory
    cov_matrix = _compute_covariance_matrix(tfbm_type, H_0, lambda_0, N)
    p_value = _common_quadratic_test_subroutine(stat_matrix, test_statistic, cov_matrix, monte_carlo_steps)
    return p_value

def tamsd_test(trajectory, tfbm_type, H_0, lambda_0, lag, monte_carlo_steps=1000):
    """
    Statistical test based on time-averaged mean squared displacement (TAMSD), described in https://doi.org/10.1063/5.0044878
    Null hypothesis: trajectory is TFBM of type tfbm_type with parameters H and lambd (λ),
    Alternative hypothesis: trajectory is not TFBM of type tfbm_type with parameters H and lambd (λ)

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to be tested.
    tfbm_type : str
        The type of TFBM ('1', '2', or '3').
    H : float
        The Hurst parameter of the TFBM.
    lambd : float
        The tempering parameter of the TFBM.
    lag : int
        The time lag for the  TAMSD.
    T : float
        The time horizon of the TFBM.
    monte_carlo_steps : int, optional
        The number of Monte Carlo samples to use for estimating the p-value (default is 1000)
    """
    return _general_quadratic_form_test(_tamsd_matrix, trajectory, tfbm_type, H_0, lambda_0, lag, monte_carlo_steps)

def avcf_test(trajectory, tfbm_type, H_0, lambda_0, lag, monte_carlo_steps=1000):
    """
    Statistical test based on sample autocovariance function, described in https://doi.org/10.1063/5.0044878
    Null hypothesis: trajectory is TFBM of type tfbm_type with parameters H and lambd (λ),
    Alternative hypothesis: trajectory is not TFBM of type tfbm_type with parameters H and lambd (λ)

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to be tested.
    tfbm_type : str
        The type of TFBM ('1', '2', or '3').
    H : float
        The Hurst parameter of the TFBM.
    lambd : float
        The tempering parameter of the TFBM.
    lag : int
        The time lag for the sample autocovariance function.
    T : float
        The time horizon of the TFBM.
    monte_carlo_steps : int, optional
        The number of Monte Carlo samples to use for estimating the p-value (default is 1000)
    """
    return _general_quadratic_form_test(_acvf_matrix, trajectory, tfbm_type, H_0, lambda_0, lag, monte_carlo_steps)

def dma_test(trajectory, tfbm_type, H, lambd, lag, T, monte_carlo_steps=1000):
    """
    Statistical test based on Detrended Moving Average (DMA) statistic, described in https://doi.org/10.1063/5.0044878
    Null hypothesis: trajectory is TFBM of type tfbm_type with parameters H and lambd (λ),
    Alternative hypothesis: trajectory is not TFBM of type tfbm_type with parameters H and lambd (λ)

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to be tested.
    tfbm_type : str
        The type of TFBM ('1', '2', or '3').
    H : float
        The Hurst parameter of the TFBM.
    lambd : float
        The tempering parameter of the TFBM.
    lag : int
        The time lag for the DMA statistic.
    T : float
        The time horizon of the TFBM.
    monte_carlo_steps : int, optional
        The number of Monte Carlo samples to use for estimating the p-value (default is 1000)
    """
    N = len(trajectory)
    stat_matrix = np.identity(N - lag + 1) / (N - lag + 1)
    test_statistic = _get_dma(trajectory, lag)
    return _common_quadratic_test_subroutine(stat_matrix, test_statistic, _dma_covariance(tfbm_type, H, lambd, N, T, lag), monte_carlo_steps)
