def exercise_1(nums, value):
    if value in nums:
        return "Yes"
    else:
        return "No"


print(exercise_1([1, 2, 3, 4, 5, 6], 2))


def exercise_2(word):
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
    while index < 100:
        n_new = n + n_plus_1
        n = n_plus_1
        n_plus_1 = n_new
        print(n_new)
        index += 1


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
