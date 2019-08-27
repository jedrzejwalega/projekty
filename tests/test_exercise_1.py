import unittest
from exercise_1 import exercise_1
import numpy as np


class TestExercise1(unittest.TestCase):

    def test_value(self):
        nums = [1, 2, 3, 4, 5, 6]
        value = 2
        actual_output = exercise_1(nums, value)
        self.assertEqual(actual_output, "Yes")

    def test_value_not_present(self):
        nums = [1, 2, 3, 4, 5, 6]
        value = 8
        actual_output = exercise_1(nums, value)
        self.assertEqual(actual_output, "No")

    def test_none(self):
        nums = [1, 2, 3, 4, 5, 6]
        value = None
        actual_output = exercise_1(nums, value)
        self.assertEqual(actual_output, "No")

    def test_nan(self):
        nums = [1, 2, 3, 4, 5, 6]
        value = np.nan
        actual_output = exercise_1(nums, value)
        self.assertEqual(actual_output, "No")

    def test_string(self):
        nums = [1, 2, 3, 4, 5, 6]
        value = "as"
        actual_output = exercise_1(nums, value)
        self.assertEqual(actual_output, "No")


if __name__ == '__main__':
    unittest.main()
