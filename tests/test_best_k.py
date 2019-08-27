import numpy as np
import unittest
from best_k import best_k

# BEST_K TESTY:


class TestBestK(unittest.TestCase):

    def test_value(self):
        nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        expected_value = 45
        actual_value = best_k(nums)
        self.assertEqual(actual_value, expected_value)

    def test_negative_value(self):
        nums = [-5, 2, 8, -4]
        expected_value = 10
        actual_value = best_k(nums)
        self.assertEqual(actual_value, expected_value)

    def test_nan(self):
        nums = [1, np.nan, 8]
        expected_value = "Error: Function doesn't accept NaN values."
        actual_value = best_k(nums)
        self.assertEqual(actual_value, expected_value)

    def test_none(self):
        nums = [1, None, 8]
        expected_value = "Error: Function doesn't accept None as a value."
        actual_value = best_k(nums)
        self.assertEqual(actual_value, expected_value)

    def test_empty_list(self):
        nums = []
        expected_value = "Error: Empty list."
        actual_value = best_k(nums)
        self.assertEqual(actual_value, expected_value)

    def test_floats(self):
        nums = [1.25, 8, 3.25]
        expected_value = 12.5
        actual_value = best_k(nums)
        self.assertEqual(actual_value, expected_value)

    def test_strings(self):
        nums = ["pies", "wes", "szmes"]
        expected_value = "Error: Function doesn't accept strings."
        actual_value = best_k(nums)
        self.assertEqual(actual_value, expected_value)


if __name__ == '__main__':
    unittest.main()
