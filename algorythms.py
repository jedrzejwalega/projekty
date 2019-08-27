def exercise_1(nums, value):
    if value in nums:
        return "Yes"
    else:
        return "No"


print(exercise_1([1, 2, 3, 4, 5, 6], 2))


def exercise_2(word):
    if type(word) is int:
        return "Error: function doesn't accept strings"
    palindrome_candidate = ""
    for index in range(len(word)):
        if index == 0:
            continue
        palindrome_candidate += word[-index]
    palindrome_candidate += word[0]
    if palindrome_candidate == word:
        return "Yes"
    else:
        return "No"


print(exercise_2("oko"))


def exercise_4():
    n = 1
    n_plus_1 = 1
    index = 2
    lst = [1, 1]
    while index < 100:
        n_new = n + n_plus_1
        n = n_plus_1
        n_plus_1 = n_new
        lst.append(n_new)
        index += 1
    return lst


print(exercise_4())


def exercise_6(array, value):
    length = len(array)
    half_length = length // 2
    while array[half_length] != value:
        if array[half_length] < value:
            half_length = half_length + len(array[half_length: -1]) // 2
        if array[half_length] > value:
            half_length = half_length - len(array[0, half_length]) // 2
        if array[half_length] == value:
            return "Yes"


exercise_6([1, 2, 3, 4, 5, 6, 7, 8, 9], 8)


def exercise_3(nums1, nums2):
    index1 = 0
    index2 = 0
    sorted_list = []

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


print(exercise_3([1, 3, 8], [1, 1, 1, 2, 4]))

def exercise_5(number):
    lst = []
    while number != 0:
        number_digit = number % 10
        lst.append(number_digit)
        number = number // 10
    reverse_lst = []
    for index in range(1, len(lst) + 1):
        reverse_lst.append(lst[-index])

    return reverse_lst

print(exercise_5(85023))
