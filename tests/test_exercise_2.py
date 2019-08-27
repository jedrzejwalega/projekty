import unittest
from exercise_2 import exercise_2
import numpy as np

class TestExercise2(unittest.TestCase):

    def test_value_positive(self):
        word = "oko"
        actual_output = exercise_2(word)
        self.assertEqual(actual_output, "Yes")

    def test_value_negative(self):
        word = "pies"
        actual_output = exercise_2(word)
        self.assertEqual(actual_output, "No")

    def test_integrers(self):
        word = 12
        actual_output = exercise_2(word)
        self.assertEqual(actual_output, "Error: function doesn't accept integrers")

    def test_nan(self):
        word = np.nan
        actual_output = exercise_2(word)
        self.assertEqual(actual_output, "Error: function doesn't accept floats or NaN")

    def test_one_letter(self):
        word = "a"
        actual_output = exercise_2(word)
        self.assertEqual(actual_output, "Error: single letter is not a word")

if __name__ == '__main__':
    unittest.main()
