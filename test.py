import numpy as np
from collections import namedtuple
import unittest
from biggestthreek import funkcja

class TestFunction(unittest.TestCase):
    def test1(self):
        list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        k = 5
        result = funkcja(list, k).biggest
        self.assertEqual(result, 40)

    def test2(self):
        list = [1, 2, 3]
        k = 2
        result = funkcja(list, k)
        self.assertEqual(result, "Error: Elements in the list are not divideable by 2.")

    def test3(self):
        list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        k  = 4
        result = funkcja(list, k)
        self.assertEqual(result, "Error: Elements in the list are not divideable by 4.")
    
    def test4(self):
        list = []
        k = 2
        result = funkcja(list, k)
        self.assertEqual(result, "Error: Empty list.")
    
    def test5(self):
        list = ["pies", "wes", "szmes"]
        k = 3
        result = funkcja(list, k)
        self.assertEqual(result, "Error: Function doesn't accept strings.")
    
    def test6(self):
        list = [1, 2, 3]
        k = 10
        result = funkcja(list, k)
        self.assertEqual(result, "Error: Index out of range.")
    
    def test7(self):
        list = [-1, -8, -4, -6]
        k = 2
        result = funkcja(list, k).biggest
        self.assertEqual(result, -9)

    def test8(self):
        list = [1.0, 8.0, 4, 25.75]
        k = 2
        result = funkcja(list, k).biggest
        self.assertEqual(result, 29.75)

    def test9(self):
        list = [1, 8, np.NaN, 12]
        k = 2
        result = funkcja(list, k)
        self.assertEqual(result, "Error: Function doesn't accept NaN values.")

    def test10(self):
        list = [1, None, 9, 12]
        k = 2
        result = funkcja(list, k)
        self.assertEqual(result, "Error: Function doesn't accept None as a value.")

    def test11(self):
        list = [None, 1, 2, 3]
        k = 2
        result = funkcja(list, k)
        self.assertEqual(result, "Error: Function doesn't accept None as a value.")


if __name__ == '__main__':
    unittest.main()

