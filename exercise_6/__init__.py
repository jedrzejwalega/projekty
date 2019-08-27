def exercise_6(array, value):
    length = len(array)
    half_length = length // 2
    for num in array:
        if type(num) is str:
            return "Error: function cannot accept strings"
    if array[half_length] == value:
            return "Yes"
    while array[half_length] != value:
        if array[half_length] < value:
            if len(array[half_length: -1]) // 2 < 1:
                half_length = half_length + 1
            else:
                half_length = half_length + len(array[half_length: -1]) // 2
        if array[half_length] > value:
            if len(array[half_length: -1]) // 2 < 1:
                half_length = half_length - 1
            else:
                half_length = half_length - len(array[half_length: -1]) // 2
        if array[half_length] == value:
            return "Yes"
        if half_length == len(array) - 1:
            return "No"
        if half_length == 0:
            return "No"