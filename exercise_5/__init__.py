def exercise_5(number: int):
    lst = []
    if type(number) is str:
        return "Error: function cannot accept strings"
    if number < 0:
        number = number * -1
    while number != 0:
        number_digit = number % 10
        lst.append(number_digit)
        number = number // 10
    reverse_lst = []
    for index in range(1, len(lst) + 1):
        reverse_lst.append(lst[-index])

    return reverse_lst
