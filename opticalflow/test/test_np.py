"""System modules"""
import numpy as np
import unittest

class TestNumpy(unittest.TestCase):

    def test_find_magnitude(self):
        arr_1 = np.array([[3.0, 4.0], [-5.0, -12.0]])
        norm_1 = np.linalg.norm(arr_1, ord=None, axis=1)
        assert np.array_equal(norm_1, np.array([5.0, 13.0])) is True

        arr_2 = np.array([[-6.0, 8.0], [7.0, -24.0]])
        norm_2 = np.linalg.norm(arr_2, ord=None, axis=1)
        assert np.array_equal(norm_2, np.array([10.0, 25.0])) is True

    def test_find_atan(self):
        arr = np.array([[1.0,1.0], [-1.0,1.0], [-1.0, -1.0], [1.0, -1.0]])
        x_arr = arr[:,0]
        y_arr = arr[:,1]
        angle = np.arctan2(y_arr, x_arr) * 180 / np.pi
        assert np.array_equal(angle, np.array([45.0, 135.0, -135.0, -45.0])) is True

    def test_horizontal_stack(self):
        x_arr = np.array([1,2,3,4])
        y_arr = np.array([-1,-2,-3,-4])
        arr = np.stack((x_arr, y_arr), axis=1)
        assert np.array_equal(arr, np.array([[1,-1],[2,-2],[3,-3],[4,-4]])) is True



if __name__ == '__main__':
    unittest.main()
