import unittest
from exercise_4 import exercise_4
import numpy as np


class TestExercise1(unittest.TestCase):

    def test_length(self):
        actual_output = len(exercise_4())
        self.assertEqual(actual_output, 100)

if __name__ == '__main__':
    unittest.main()

