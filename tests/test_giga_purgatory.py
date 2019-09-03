import unittest
from giga_purgatory import all_combinations
import numpy as np


class TestAllCombinations(unittest.TestCase):

    def test_n_one(self):
        actual_value = all_combinations(1)
        expected_value = ["0", "1"]
        self.assertEqual(actual_value, expected_value)

    def test_n_three(self):
        actual_value = all_combinations(3)
        expected_value = [["0", "0", "0"], ["0", "0", "1"], ["0", "1", "0"], ["0", "1", "1"], ["1", "0", "0"], ["1", "0", "1"], ["1", "1", "0"], ["1", "1", "1"]]
        self.assertListEqual(actual_value, expected_value)
    
    def test_zero(self):
        actual_value = all_combinations(0)
        expected_value = "Error: n must be bigger or equal to 1"
        self.assertEqual(actual_value, expected_value)

    def test_float(self):
        actual_value = all_combinations(3.0)
        expected_value = [["0", "0", "0"], ["0", "0", "1"], ["0", "1", "0"], ["0", "1", "1"], ["1", "0", "0"], ["1", "0", "1"], ["1", "1", "0"], ["1", "1", "1"]]
        self.assertListEqual(actual_value, expected_value)

    def test_list(self):
        actual_value = all_combinations([1, 2, 3])
        expected_value = "Error: function only accepts integrers and floats"
        self.assertEqual(actual_value, expected_value)
    
    def test_none(self):
        actual_value = all_combinations(None)
        expected_value = "Error: function only accepts integrers and floats"
        self.assertEqual(actual_value, expected_value)
    
    def test_none(self):
        actual_value = all_combinations(np.nan)
        expected_value = "Error: function doesn't accept NaN."
        self.assertEqual(actual_value, expected_value)

if __name__ == '__main__':
    unittest.main()
