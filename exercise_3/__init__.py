import numpy as np


def exercise_3(nums1, nums2):
    index1 = 0
    index2 = 0
    sorted_list = []

    # Kinda beats the point of the exercise, but somehow I have to check for strings, NaNs etc.
    for num1 in nums1:
        if type(num1) == str:
            return "Error: cannot accept string"
        if num1 is None:
            return "Error: cannot accept None"

    for num2 in nums2:
        if type(num2) == str:
            return "Error: cannot accept string"
        if num2 == None:
            return "Error: cannot accept None"

    if np.isnan(nums1).any() == True:
        return "Error: cannot accept NaN"

    if np.isnan(nums2).any() == True:
        return "Error: cannot accept NaN"

    # Compares positions at each list, until one of them runs out of values
    while index1 <= len(nums1) - 1 and index2 <= len(nums2) - 1:
        if nums1[index1] <= nums2[index2]:
            sorted_list.append(nums1[index1])
            index1 += 1
        else:
            sorted_list.append(nums2[index2])
            index2 += 1

    # When first list runs out of values, the function adds the rest of the numbers in the second list to the sorted list
    if index1 > len(nums1) - 1:
        while index2 <= len(nums2) - 1:
            sorted_list.append(nums2[index2])
            index2 += 1

    # When second list runs out of values, the function adds the rest of the numbers in the firstlist to the sorted list
    if index2 > len(nums2) - 1:
        while index1 <= len(nums1) - 1:
            sorted_list.append(nums1[index1])
            index1 += 1
    return sorted_list
