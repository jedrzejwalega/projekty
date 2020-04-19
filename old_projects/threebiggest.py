import numpy as np


def funkcja(list):
    list_length = len(list)
    list_index = range(len(list))

    if list_length == 0:
        return "Error: Empty list"
    if list_length % 3 != 0:
        return "Error: Elements in the list are not divideable by 3."

    for n in list_index:

        if type(list[n]) is str:
            return "Error: Function doesn't accept strings."
        if list_length - n > 2:
            biggest = list[n] + list[n + 1] + list[n + 2]
            if n == 0:
                biggest_new = biggest
            else:
                if biggest > biggest_new:
                    biggest_new = biggest
        else:
            continue
    if np.isnan(list).any() == True:
        return "Error: Function doesn't accept NaN values."
    return biggest_new


def test(nums, expected):
    result = funkcja(nums)
    if result == expected:
        print("PASS")
    else:
        print("FAIL - list = {list}, expected output = {expected}, actual output = {result}".format(list=nums, expected=expected, result=result))


def funkcja_test():
    test([1, 2, 3, 4, 5, 6], 15)
    test([], "Error: Empty list")
    test([1], "Error: Elements in the list are not divideable by 3.")
    test([1, 2], "Error: Elements in the list are not divideable by 3.")
    test([1, 2, 3, 4], "Error: Elements in the list are not divideable by 3.")
    test([1, 2, 3, 4, 5], "Error: Elements in the list are not divideable by 3.")
    test([1, np.NaN, 8], "Error: Function doesn't accept NaN values.")
    test(["echo", "wow", "lol"], "Error: Function doesn't accept strings.")
    test([-8, -2, -24, -2, -1, -5], -8)
    test([8.0, 2, 34, 25, 26.8, 12.9], 85.8)


funkcja_test()
