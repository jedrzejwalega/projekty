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