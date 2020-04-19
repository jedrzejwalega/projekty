import numpy as np


# BEST_K FUNKCJA

def best_k(nums):
    biggest = 0
    for k in range(0, len(nums) + 1):
        new_biggest = funkcja(nums, k)
        if biggest == 0:
            biggest = new_biggest
        else:
            biggest = max(biggest, new_biggest)
        if type(new_biggest) is str:
            return new_biggest
    return biggest


# FUNKCJA FUNKCJA

def funkcja(nums: list, k: int):

    list_length = len(nums)
    k_index = range(k)

    # checking if the list is alright

    if list_length == 0:
        return "Error: Empty list."
    if list_length < k:
        return "Error: Index out of range."
    for num in nums:
        if num is None:
            return "Error: Function doesn't accept None as a value."
        if type(num) is str:
            return "Error: Function doesn't accept strings."

    # calculating biggest value

    for n in range(list_length):

        if list_length - n >= k:
            k_counter = 0  # prevents the for loop from adding more values than stated by k
            biggest = 0
            while k_counter < k:
                for k_part in k_index:
                    biggest += nums[n + k_part]
                    k_counter += 1

            if n == 0:
                biggest_new = biggest
            else:
                biggest_new = max(biggest, biggest_new)
        else:
            continue
    if np.isnan(nums).any() == True:
        return "Error: Function doesn't accept NaN values."
    return biggest_new
