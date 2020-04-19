import unittest
from exercise_3 import exercise_3
import numpy as np


class TestExercise3(unittest.TestCase):

    def test_value(self):
        nums1 = [1, 3, 8]
        nums2 = [1, 1, 1, 2, 4]
        actual_output = exercise_3(nums1, nums2)
        self.assertEqual(actual_output, [1, 1, 1, 1, 2, 3, 4, 8])

    def test_string(self):
        nums1 = [1, 3, "dog"]
        nums2 = [1, 1, 1, 2, 4]
        actual_output = exercise_3(nums1, nums2)
        self.assertEqual(actual_output, "Error: cannot accept string")

    def test_string2(self):
        nums1 = [1, 3, 4]
        nums2 = [1, 1, 1, 2, "dog"]
        actual_output = exercise_3(nums1, nums2)
        self.assertEqual(actual_output, "Error: cannot accept string")

    def test_nan(self):
        nums1 = [1, 3, 4]
        nums2 = [1, 1, 1, np.nan, 7]
        actual_output = exercise_3(nums1, nums2)
        self.assertEqual(actual_output, "Error: cannot accept NaN")

    def test_none(self):
        nums1 = [1, 3, 4]
        nums2 = [1, 1, 1, None, 7]
        actual_output = exercise_3(nums1, nums2)
        self.assertEqual(actual_output, "Error: cannot accept None")

    def test_none2(self):
        nums1 = [1, 3, None]
        nums2 = [1, 1, 1, 4, 7]
        actual_output = exercise_3(nums1, nums2)
        self.assertEqual(actual_output, "Error: cannot accept None")

    def test_empty(self):
        nums1 = [1, 3, 5]
        nums2 = []
        actual_output = exercise_3(nums1, nums2)
        self.assertEqual(actual_output, [1, 3, 5])

    def test_empty2(self):
        nums1 = []
        nums2 = [2, 4, 7]
        actual_output = exercise_3(nums1, nums2)
        self.assertEqual(actual_output, [2, 4, 7])

    def test_float(self):
        nums1 = [4, 5.25, 8]
        nums2 = [3, 5, 19]
        actual_output = exercise_3(nums1, nums2)
        self.assertEqual(actual_output, [3, 4, 5, 5.25, 8, 19])


if __name__ == '__main__':
    unittest.main()
