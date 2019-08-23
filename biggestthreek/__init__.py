import numpy as np
from collections import namedtuple


def funkcja(list, k):
    list_length = len(list)
    list_index = range(len(list))
    k_index = range(k)
    if list_length == 0:
        return "Error: Empty list."
    if list_length < k:
        return "Error: Index out of range."
    if list_length % len(k_index) != 0:
        return "Error: Elements in the list are not divideable by {k_index}.".format(k_index=len(k_index))

    for n in list_index:

        if type(list[n]) is str:
            return "Error: Function doesn't accept strings."

        if list_length - n >= len(k_index):
            k_counter = 0
            biggest = 0
            while k_counter < len(k_index):
                for k_part in k_index:
                    if list[n + k_part] is None:
                        return "Error: Function doesn't accept None as a value."
                    else:
                        biggest += list[n + k_part]

                        k_counter += 1

            if n == 0:
                biggest_new = biggest
            else:
                if biggest > biggest_new:
                    biggest_new = biggest
        else:
            continue
    if np.isnan(list).any() is True:
        return "Error: Function doesn't accept NaN values."
    output = namedtuple("output", ["list", "list_length", "biggest"])
    parameters = output(list, list_length, biggest_new)
    return parameters
