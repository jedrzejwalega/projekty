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
