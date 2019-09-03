def version_2(options):
    new_options = []
    for option in options:
        for i in range(len(option)):
            new_option = list(option)
            new_option[i] = "+"
            new_option_str = "".join(new_option)
            if new_option_str not in new_options:
                new_options.append(new_option_str)
            else:
                continue
    return new_options


def power(n):
    final = ""
    counter = 0
    options = []
    while counter < n:
        final += "-"
        counter += 1
    options.append(final)
    for x in range(n):
        options = version_2(options)
    options = options + [final]
    return options



for i in range(6):
    sor = (sorted(power(i)))
    sor.reverse()
    print(sor)
    print(len(sor))
    print("\n")

