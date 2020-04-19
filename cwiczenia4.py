import numpy as np

def one():
    lst = eval(input("Enter a list of integrers: "))
    print(len(lst))
    print(lst[-1])
    lst_copy = lst[:]
    lst_copy.reverse()
    print(lst_copy)
    if 5 in lst:
        print("Yes")
    else:
        print("No")
    print(lst.count(5))
    lst.pop(-1)
    lst.pop(0)
    lst.sort()
    counter = 0
    for element in lst:
        if element < 5:
            counter += 1
    print(counter)
# print(one())

def two():
    randoms = []
    for n in range(20):
        randoms.append(np.random.randint(1, 100))
    print(randoms)
    average = sum(randoms) / len(randoms)
    print(average)
    print(max(randoms))
    print(min(randoms))
    randoms.sort()
    print("Second smallest ", randoms[1], "\n" + "Second biggest: ", randoms[-2])
    counter = 0
    for element in randoms:
        if element % 2 == 0:
            counter += 1
    print(counter)
    
# print(two())

def three():
    lst = [8, 9, 10]
    lst[1] = 17
    lst.append(4)
    lst.append(5)
    lst.append(6)
    print(lst)
    lst.pop(0)
    print(lst)
    lst.sort()
    print(lst)
    lst2 = lst * 2
    print(lst2)
    lst2.insert(3, 25)
    print(lst2)
# print(three())

def four():
    lst = eval(input("Enter a list containing numbers between 1 and 12: "))
    for i in range(len(lst)):
        if lst[i] > 10:
            lst[i] = 10
    return lst
# print(four())

def five():
    lst = eval(input("Enter a list of strings: "))

    print(lst)
    print(type(lst))
    new_lst = []
    for string in lst:
        print(string)
        new_lst.append(string[1:])
    return new_lst
# print(five())

def six():
    lst_one = []
    for i in range(50):
        lst_one.append(i)

    lst_two = []
    for i in range(51):
        lst_two.append(i**2)
    
    lst_three = []
    counter = 1
    for letter in "abcdefghijklmnopqrstuvwxyz":
        lst_three.append(letter * counter)
        counter += 1
    print(lst_three)

# print(six())
    
def seven():
    lst_1 = eval(input("Enter list 1: "))
    lst_2 = eval(input("Enter list 2: "))
    lst_3 = []
    for i in range(len(lst_1)):
        lst_3.append(lst_1[i] + lst_2[i])
    return lst_3
print(seven())

def ten():
    lst = [0, 1, 2, 3, 4, 5, 6]
    for i in range(len(lst)):
        
