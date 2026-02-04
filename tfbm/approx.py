import numpy as np

def genereate_circulant_row(cov_fun, embed_exp:int, sample_size:int):
    # Create first row of circulant matrix as descibed in equations (2.2)-(2.3) in Wood-Chan paper https://doi.org/10.1080/10618600.1994.10474655
    m =  2 ** embed_exp
    times = np.concatenate([np.arange(start = 0, stop=m//2), np.arange(start=m//2, stop=0, step=-1)]) / sample_size
    matrix_row = cov_fun(times)
    return matrix_row

def calculate_eigen_vals(cov_fun, embed_exp:int, sample_size:int):
    return np.fft.fft(cov_fun, embed_exp, sample_size).real

def wood_chan_increments(m, sample_size, eigenvals):
    U = np.random.normal(0, 1, m - 1)
    V =  np.random.normal(0, 1, m - 1)

    a = np.zeros(2*m, dtype=complex)
    a[1:(m//2)] = (U + 1j * V) / np.sqrt(2)
    a[1+m//2:] = np.conj(a[1:(m//2)][::-1])
    a[0] = np.random.normal(0, 1)
    a[m//2] =  np.random.normal(0, 1)
    
    a = np.sqrt(eigenvals) * a

    X = np.real(np.fft.fft(a) / np.sqrt(m))[:sample_size]
    return X

def find_optimal_eigenvals(cov_fun, max_embedding_exp:int, sample_size:int, approximate:bool):
    embed_exp = np.ceil(np.log2(sample_size-1))
    while embed_exp <= max_embedding_exp:
        eigen_vals = calculate_eigen_vals(cov_fun, embed_exp, sample_size)
        if np.all(eigen_vals >= 0):
            break
    
    if embed_exp > max_embedding_exp:
        if approximate:
            positive_eigen_vals = np.maximum(eigen_vals, 0)
            scale_factor = np.sqrt(np.sum(eigen_vals) / np.sum(positive_eigen_vals))
            eigen_vals = positive_eigen_vals * scale_factor
            print("Warning: Maximum embedding exponent exceeded. Using adjusted eigenvalues.")
        else:
            raise ValueError("Wood-Chan: For maximal embedding circulant matris is not semi-positive")
    return int(embed_exp), eigen_vals

def wood_chan(cov_fun, max_embedding_exp:int, sample_size:int):
    embed_exp, eigen_vals = find_optimal_eigenvals(cov_fun, max_embedding_exp, sample_size)
    m = 2 ** embed_exp
    increments = wood_chan_increments(m, sample_size, eigen_vals)
    return np.cumsum(increments), increments