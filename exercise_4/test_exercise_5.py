import unittest
from exercise_5 import exercise_5
import numpy as np


class TestExercise5(unittest.TestCase):

def test_value(self):
    actual_value = exercise_5(8542)
    expected_value = [8, 5, 4, 2]
    self.assertEqual(actual_value, expected_value)

if __name__ == '__main__':
    unittest.main()
