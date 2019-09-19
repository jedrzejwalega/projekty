import unittest
from exercise_5 import exercise_5
import numpy as np


class TestExercise5(unittest.TestCase):

    def test_value(self):
        actual_value = exercise_5(8542)
        expected_value = [8, 5, 4, 2]
        self.assertEqual(actual_value, expected_value)

    def test_negative_number(self):
        actual_value = exercise_5(-8542)
        expected_value = [8, 5, 4, 2]
        self.assertEqual(actual_value, expected_value)

    def test_string(self):
        actual_value = exercise_5("pies")
        expected_value = "Error: function cannot accept strings"
        self.assertEqual(actual_value, expected_value)
    
    def test_float(self):
        actual_value = exercise_5(8542.25)
        expected_value = [8, 5, 4, 2, 2, 5]
        self.assertEqual(actual_value, expected_value)

if __name__ == '__main__':
    unittest.main()
