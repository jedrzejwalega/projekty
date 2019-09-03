def mini_power(n):
    new_options = []
    first = ""
    for x in range(n):
        first += "-"
    new_options.append(first)
    for option in new_options:
        for i in range(n):
            new_option = list(option)
            new_option[i] = "+"
            new_option_str = "".join(new_option)
            if new_option_str not in new_options:
                new_options.append(new_option_str)
            else:
                continue
    return new_options


options = mini_power(3)
print(options)

print(mini_power(5))
