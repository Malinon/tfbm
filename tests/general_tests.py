import pytest
import numpy as np
from tfbm import TFBM1, TFBM2, TFBM3
import os
import glob

@pytest.fixture(autouse=True)
def setup_module():
    files_in_dir = [glob.glob('cov_matrices_tfbm{}/*.txt'.format(i)) for i in range(1, 4)]
    for files in files_in_dir:
        for f in files:
            os.remove(f)

tfbm_generators = [TFBM1, TFBM2, TFBM3]
tfbm_parameters = [(100, 10, 0.7, 0.3), (50, 112, 0.7, 0.3), (1, 900, 0.7, 0.3), (10, 100, 0.9, 2.0), (1, 100, 0.5, 0.5)]
tfbm_param_with_extreme = [(1, 900, 0.3, 0.7), (50, 50, 2.3, 0.5), (40, 30, 3.3, 2.5)]
methods = ["davies-harte", "cholesky"]

@pytest.mark.parametrize("tfbm_generator", tfbm_generators)
@pytest.mark.parametrize("T, N, H, lambd", tfbm_parameters)
@pytest.mark.parametrize("method", methods)
def test_shape_and_content(tfbm_generator, T, N, H, lambd, method):
    gen = tfbm_generator(T, N, H, lambd, method)
    out_333 = gen.generate_samples(333)

    assert out_333.shape == (333, N+1)
    assert out_333.dtype == float
    assert not np.isnan(out_333).any()
    assert not np.isinf(out_333).any()
    assert gen.generate_samples(40).shape == (40, N+1)


@pytest.mark.parametrize("tfbm_generator", tfbm_generators[:2])
@pytest.mark.parametrize("T, N, H, lambd", tfbm_param_with_extreme)
@pytest.mark.parametrize("method", methods)
def test_extreme(tfbm_generator, T, N, H, lambd, method):
    gen = tfbm_generator(T, N, H, lambd, method)
    out_333 = gen.generate_samples(333)

    assert out_333.shape == (333, N+1)
    assert out_333.dtype == float
    assert not np.isnan(out_333).any()
    assert not np.isinf(out_333).any()

@pytest.mark.parametrize("tfbm_generator", tfbm_generators)
def test_starts_with_zero(tfbm_generator):
    gen = tfbm_generator(1, 100, 0.55, 0.1)
    assert (gen.generate_samples(100)[:, 0] == 0).all()

@pytest.mark.parametrize("tfbm_generator", tfbm_generators)
def test_covariance_matrix_ok(tfbm_generator):
    sigma =  tfbm_generator(1, 100, 0.55, 0.1).covariance_matrix()
    assert not np.isnan(sigma).any()
    assert not np.isinf(sigma).any()
    assert np.allclose(sigma, sigma.T)



