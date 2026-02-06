import numpy as np

def _get_dma(trajectory, lag):
    trajectory_extended = np.insert(trajectory, 0, 0, axis=0)
    trajectory_cumsum = np.cumsum(trajectory_extended, axis=0)
    moving_avg = (trajectory_cumsum[lag:] - trajectory_cumsum[:-lag]) / lag
    detrended_trajectory = trajectory[(lag - 1):] - moving_avg
    print(detrended_trajectory)
    dma = np.mean(detrended_trajectory**2)
    return dma

def _dma_covariance(N, lag):
    basic_cov = np.random.rand(N,N)
    basic_cov[1,3] = 4.63
    basic_cov += basic_cov.transpose()
    print(basic_cov)
    dma_covariance = np.zeros((N - lag + 1, N - lag + 1))
    # for j in range(N):
    #     for k in range(j+1):
    #         dma_covariance[j, k-1] = ( ((1 - 1/lag)**2) * basic_cov[j, k]
    #         + (1/(lag**2) - 1 / lag) * basic_cov[j + lag - 1, k:(k+lag-1)].sum()
    #         + (1/(lag**2) - 1 / lag) * basic_cov[k + lag - 1, j:(j+lag-1)].sum()
    #         + (1/(lag**2)) * basic_cov[j:(j+lag-1), k:(k+lag-1)].sum())
    for j in range(N-lag+1):
        for k in range(j+1):
            dma_covariance[j, k] = ( ((1 - 1/lag)**2) * basic_cov[j+lag-1, k+lag-1]
            + (1/(lag**2) - 1 / lag) * basic_cov[j + lag - 1, k:(k+lag-2)].sum()
            + (1/(lag**2) - 1 / lag) * basic_cov[k + lag - 1, j:(j+lag-2)].sum()
            + (1/(lag**2)) * basic_cov[j:(j+lag-2), k:(k+lag-2)].sum())
            dma_covariance[k, j] = dma_covariance[j, k]
    return dma_covariance

if __name__ == '__main__':
    # _get_dma(np.array([1,2,3,4,5,6]),2)
    print(_dma_covariance(4,2))
    