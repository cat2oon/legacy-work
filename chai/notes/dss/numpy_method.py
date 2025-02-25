import unittest
import numpy as np

class TestNumpyMethod(unittest.TestCase):
    
    
    
    """
        Test Helper
    """
    def assert_array(self, actual, expected):
        self.assertTrue(np.allclose(actual, expected))

        
        
    """
        Generator
    """
    def test_arange(self):
        n = 5
        self.assert_array(np.arange(n), [0,1,2,3,n-1])
        self.assert_array(np.arange(10), [0,1,2,3,4,5,6,7,8,9])
        
    def test_zeros(self):
        expected = [[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]
        self.assert_array(np.zeros((3,5)), expected)
        
    def test_ones(self):
        expected = [[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1], 
                    [1, 1, 1], 
                    [1, 1, 1]]
        self.assert_array(np.ones((5, 3)), expected)
        
    def test_diag(self):
        actual = np.diag(np.arange(4))
        expected = [[0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 2, 0],
                    [0, 0, 0, 3]]
        self.assert_array(actual, expected)
        
    def test_identity(self):
        actual = np.identity(4)
        expected = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]]
        self.assert_array(actual, expected)
        
    def test_eye(self):
        actual = np.eye(3)
        expected = [[1, 0, 0,],
                    [0, 1, 0,],
                    [0, 0, 1,]]
        self.assert_array(actual, expected)
        
    def test_eye_equal_identity(self):
        actual = np.eye(5)
        expected = np.identity(5)
        self.assert_array(actual, expected)   
        
        
    
    """
        Mutator
    """
    def test_T(self):
        arr = np.array([[1, 2, 3], 
                        [4, 5, 6]])
        expected = [[1, 4], 
                    [2, 5], 
                    [3, 6]]
        self.assert_array(arr.T, expected)   
        self.assert_array(arr, arr.T.T)   

    def test_T_with_square_mat(self):
        arr = np.array([[1, 2, 3], 
                        [4, 5, 6], 
                        [7, 8, 9]])
        expected = [[1, 4, 7], 
                    [2, 5, 8], 
                    [3, 6, 9]]
        self.assert_array(arr.T, expected)   
        self.assert_array(arr, arr.T.T)   
    
    def test_T_1d(self):
        arr = np.array([1, 2, 3])
        
        self.assertTrue(arr.shape == (3,))
        self.assert_array(arr.T, arr)   
        self.assert_array(arr, arr.T.T)   
    
    
    
    
    """ 
        Characterisitc
    """
    def test_mean_removed_broadcasting(self):
        """ Mean Revmoed == zero-mean """
        arr = np.array([1,2,3,4,5,6,7])  # range(1,8)
        mean = arr.mean()           # flaot
        
        mean_removed = arr - mean
        expected = np.array([-3,-2,-1,0,1,2,3])
        
        self.assertTrue(mean.dtype == np.float64)
        self.assert_array(mean_removed, expected)
        
        
        
    """ 
        Operator
    """ 
    def test_mean(self):
        arr = np.array(range(1, 11))
        self.assertTrue(arr.mean() == 5.5)
        
    def test_sum(self):
        arr = np.array(range(1, 11))
        self.assertTrue(arr.sum() == 55)
        
    def test_inner_dot_product_between_vector(self):
        v1 = np.array([1,2,3,4])
        v2 = np.array([4,5,6,7])
        expected = 1*4 + 2*5 + 3*6 + 4*7
        actual = v1 @ v2        
        dot = np.dot(v1, v2)
        
        self.assertTrue(v1.shape[0] == v2.shape[0])
        self.assertTrue(actual.dtype == np.int64)
        self.assert_array(actual, expected)
        self.assert_array(dot, expected)
        
    def test_weighted_sum(self):
        price = np.array([100.0, 80.0, 50.0])
        num = np.array([3, 4, 5])
        self.assert_array(price@num, 870.0)
    
    def test_weighted_average(self):
        score = np.array([100, 60])
        weight = np.array([1, 3])
        w_sum = score @ weight
        actual = w_sum / weight.sum()
        self.assertTrue(actual, 70)
        
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        