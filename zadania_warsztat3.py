
def number_one():
    user_inp = str(input("Please enter the string, doesn't have to be in ticks: "))
    inp_len = len(user_inp)
    ten_times = user_inp * 10
    first_character = user_inp[0]
    first_three = user_inp[:3]
    last_three = user_inp[-3:]
    backwards = ""
    for index in range(1, len(user_inp) + 1):
        backwards += user_inp[-index]

    if len(user_inp) >= 7:
        message = user_inp[6]
    else:
        message = "Error: string is not long enough"
    
    removed = user_inp[1:-1]
    all_caps = user_inp.upper()
    chapter = user_inp.replace("a", "CHAPTER")
    space = ""
    for index in range(len(user_inp)):
        new = user_inp[index] + " "
        space += new
    space2 = ""
    for index in range(len(user_inp)):
        space2 += " "
    space2 += "tutaj sa spacje jbc"
        
    return inp_len, ten_times, first_character, first_three, last_three, backwards, message, removed, all_caps, chapter, space, space2

def number_two():
    user_inp = str(input("GIMME A STRING, MAN: "))
    number_of_words = user_inp.count(" ")
    return number_of_words

def number_three():
    user_input = str(input("Gimme a formula: "))
    first_count = user_input.count(")")
    second_count = user_input.count("(")
    if first_count == 0 and second_count == 0:
        return ("There are no ( or ) in your formula")
    if first_count == second_count:
        return("The number is even")
    else:
        return("The number is not even")

def number_four():
    user_input = str(input("STRING, NOW: "))
    vowels = "aeiouy"
    for letter in vowels:
        if letter in user_input:
            return True

def number_five():
    user_input = str(input("STRING, NOW: "))
    new_string = ""
    for index in range(len(user_input)):
        if index == 1:
            new_string += "*"
        new_string += user_input[index]
    new_string += "!!!"
    return new_string

def number_six():
    s = str(input("STRING, NOW: "))
    new_s = ""
    for index in range(len(s)):
        if s[index] == "," or s[index] == "-":
            continue
        else:
            new_s += s[index]
    new_s = new_s.lower()
    return new_s

def number_seven():
    user_input = str(input("STRING, NOW: "))
    backwards = ""
    for index in range(1, len(user_input) + 1):
        backwards += user_input[-index]
    if backwards == user_input:
        return "The word is a palindrome"
    else:
        return "The word is not a palindrome"

def number_eight():
    number_of_adresses = eval(input("Enter the number of addresses: "))
    addresses = []
    for n in range(number_of_adresses):
        a = n + 1
        addresses.append(str(input("Write the address {a}: ".format(a = a))))
    print(addresses)
    print(addresses.count("@prof."))
    if addresses.count("@prof.") == len(addresses):
        return "All addresses belong to professors"
    if addresses.count("@student.") == len(addresses):
        return "All addresses belong to students"
    if addresses.count("@prof.") != len(addresses) and addresses.count("@student.") != len(addresses):
        return "The adresses are mixed"


def nine():
    user_input = eval(input("Number: "))
    a = 0
    while a != user_input:
        print(" " * a + str(a + 1))
        a += 1

def ten():
    user_input = str(input("String: "))
    for letter in user_input:
        print(letter + letter)
    

# print(number_one())
# print(number_two())
# print(number_three())
# print(number_four())(
# print(number_five())
# print(number_six())
# print(number_seven())
print(number_eight())
# print(nine())
# print(ten())