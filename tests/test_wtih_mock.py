import unittest
from unittest.mock import patch
import numpy as np
from tfbm import TFBM1, TFBM2, TFBM3


class TestNumpyPatching(unittest.TestCase):
    
    @patch('numpy.random.normal')
    def test_patch_zero_randoms_result_in_zero_trajectories(self, mock_rand):
        """Test patching numpy.random.rand function"""
        # Set up mock return value
        def side_effect_func(*args, **kwargs): 
            if len(args) == 3:
                return np.array([0.0] * args[2])
            else:
                return 0.0
        
        mock_rand.side_effect = side_effect_func
        for tfbm_type in [TFBM1, TFBM2, TFBM3]:
            trajs = tfbm_type(1, 100, 0.55, 0.1).generate_samples(50)
            np.testing.assert_array_equal(trajs, np.zeros((50, 101)))
    
    @patch('numpy.random.normal')
    def test_patch_dependence_on_ramdom_is_multiplicative(self, mock_rand):
        """Test patching numpy.random.rand function"""
        # Set up mock return value
        def side_effect_func(*args, **kwargs): 
            if len(args) == 3:
                return np.array([1.0] * args[2])
            else:
                return 1.0
        
        mock_rand.side_effect = side_effect_func
        trajs = TFBM1(1, 100, 0.55, 0.1).generate_samples(1)
        def side_effect_func_new(*args, **kwargs): 
            if len(args) == 3:
                return np.array([3.0] * args[2])
            else:
                return 3.0
        
        mock_rand.side_effect = side_effect_func_new

        trajs_twice = TFBM1(1, 100, 0.55, 0.1).generate_samples(1)
        np.testing.assert_allclose(3 * trajs, trajs_twice)

    @patch('numpy.random.normal')
    def test_patch_dependence_on_eigenvals_is_sqrt(self, mock_rand):
        size=100
        def side_effect_func(*args, **kwargs): 
            if len(args) == 3:
                return np.array([1.0] * args[2])
            else:
                return 1.0
        mock_rand.side_effect = side_effect_func
        eigenvals_one = np.array([1.0] * (size * 2))
        increments_one = TFBM1(1, size, 0.55, 0.1)._generate_dh_increments(eigenvals_one)
        eigenvals_four = np.array([4.0] * (size * 2))
        increments_four = TFBM1(1, size, 0.55, 0.1)._generate_dh_increments(eigenvals_four)
        np.testing.assert_allclose(2 * increments_one, increments_four)
    

if __name__ == '__main__':
    unittest.main()