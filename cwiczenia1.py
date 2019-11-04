# # "Practical Introduction to Python programming" - ksiazka + "Python Tutorial 3.7.0 Guido van Rossum"
# print("Hello world")
# # name = input("Enter your name: ") #NOWE
# # print("hello", name)

# # Kalkulator
# print(2 + 2)
# print(80 % 3)
# print(50 / 3)
# print(50 // 3)

# temp = eval(input(" Enter temp:")) # eval - dzieki niemu input nie jest traktowany jako string, tylko liczba

# str1 = "blazej przystajko"
# for letter in str1:
#     print(letter)

# a = 10
# b = 20
# if a is not 20:
#     print("pies")

# if 3 + 4 > 5:
#     print("good")
# else:
#     print("not good")

# a = input("podaj stringa: ")
# print(a.upper())
# print(a.split())

def name_writing():
    inp = input("Some string: ")
    return inp

# print(name_writing())

# def trojkacik():
#     a = eval(input("Podaj bok a: "))
#     b = eval(input("Podaj bok b: "))
#     c = eval(input("Podaj bok c: "))
#     if a + b > c and a + c > b and b + c > a:
#         if a == b == c:
#             return "bedzie, i to rownoboczny"
#         if a == b or a == c or b == c:
#             return "bedzie, i to rownoramienny"
#         else:
#             return "bedzie dobrze"
#     else:
#         return "nie bedzie dobrze"

# print(trojkacik())

# temp = eval(input("Podaj temp: "))
# K_temp = 273 + temp
# if temp is not 0:
#     if K_temp < 273:
#         print("zimno")
#     else:
#         print("cieplo")
# else:
#     print("trzecia zasada termodynamiki")