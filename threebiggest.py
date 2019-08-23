import numpy as np
from collections import namedtuple



def funkcja(list, k):
    list_length = len(list)
    list_index = range(len(list))
    k_index = range(k)
    if list_length == 0:
        return "Error: Empty list"
    if list_length % 3 != 0:
        return "Error: Elements in the list are not divideable by 3."

    for n in list_index:

        if type(list[n]) is str:
            return "Error: Function doesn't accept strings."
        if list_length - n > 2:
            biggest = list[n] + list[n + 1] + list [n + 2]
            if n == 0:
                biggest_new = biggest
            else:
                if biggest > biggest_new:
                    biggest_new = biggest
        else:
            continue
    if np.isnan(list).any() == True:
        return "Error: Function doesn't accept NaN values."
    output = namedtuple("output", ["list", "list_length", "biggest"])
    parameters = output(list, list_length, biggest_new)
    return parameters



def test_funkcja():
    test1 = funkcja([1, 2, 3, 4, 5, 6])
    print(test1)
    if test1.biggest == 15:
        print("PASS")
    else:
        print("FAIL") 

    test2 = funkcja([])
    if test2 == "Error: Empty list":
        print("PASS")
    else:
        print("FAIL") 

    test3 = funkcja([1]) 
    if test3 == "Error: Elements in the list are not divideable by 3.":
        print("PASS")
    else:
        print("FAIL")

    test4 = funkcja([1, 2]) 
    if test4 == "Error: Elements in the list are not divideable by 3.":
        print("PASS")
    else:
        print("FAIL")

    test5 = funkcja([1, 2, 3, 4]) 
    if test5 == "Error: Elements in the list are not divideable by 3.":
        print("PASS")
    else:
        print("FAIL")

    test6 = funkcja([1, 2, 3, 4, 5])
    if test6 == "Error: Elements in the list are not divideable by 3.":
        print("PASS")
    else:
        print("FAIL")

    test7 = funkcja([1, np.NaN, 8]) 
    if test7 == "Error: Function doesn't accept NaN values.":
        print("PASS")
    else:
        print("FAIL")

    test8 = funkcja(["echo", "wow", "lol"]) 
    if test8 == "Error: Function doesn't accept strings.":
        print("PASS")
    else:
        print("FAIL")

    test9 = funkcja([-8, -2, -24, -2, -1, -5]) 
    if test9.biggest == -8:
        print("PASS")
    else:
        print("FAIL")

    test10 = funkcja([8.0, 2, 34, 25, 26.8, 12.9]) 
    if test10.biggest == 85.8:
        print("PASS")
    else:
        print("FAIL") 

print(test_funkcja())


