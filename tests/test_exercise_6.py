import unittest
from exercise_6 import exercise_6
import numpy as np


class TestExercise1(unittest.TestCase):


    def test_value2(self):
        array = [1, 3, 8]
        value = 8
        actual_output = exercise_6(array, value)
        self.assertEqual(actual_output, "Yes")
    
    def test_negative(self):
        array = [-8, -2, 3]
        value = -8
        actual_output = exercise_6(array, value)
        self.assertEqual(actual_output, "Yes")

    def test_negative2(self):
        array = [-8, -2, 3]
        value = 8
        actual_output = exercise_6(array, value)
        self.assertEqual(actual_output, "No")
    
    def test_string(self):
        array = ["pies", "wes", "szmes"]
        value = "pies"
        actual_output = exercise_6(array, value)
        self.assertEqual(actual_output, "Error: function cannot accept strings")

if __name__ == '__main__':
    unittest.main()
