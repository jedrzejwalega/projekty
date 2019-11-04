# def silnia():
#     value = 1
#     for i in range(1, 11):
#         value = value * i
#         print(value)
#     return value

# print(silnia())

# def parzyste():
#     suma = 0
#     inp = eval(input("Input max border (included): "))
#     for number in range(0, inp + 1, 2):
#         print(number)
#         suma += number
#     return suma

# print(parzyste())

# for i in range(10, 0, -1):
#     print(i)


# for i in range(101):
#     print("Jedrzej Wojciech Krzysztof WALEGA", i)

# DO POPRAWY:
# def choinka():
#     czub = "*"
#     for i in range(10, 0, -1):
#         if i == 10:
#             print(" " * i, czub)
#         else:
#             czub = "*" + czub + "*"
#             print(" " * i, czub)
#     for i in range(10, 0, -1):
#         print("*" * (i-1), "*", "*" * (i-1), sep="")

# choinka()

def wielokrotnosci():
    for number in range(10):
        print("The number is: ", number)
        for i in range(10):
            print(number * i)

wielokrotnosci()