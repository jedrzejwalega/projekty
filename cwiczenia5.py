def potegowanie1(n, k):
    wynik = 1
    for x in range(1, k + 1):
        wynik = wynik * n
    return wynik
# print(potegowanie1(10, 3))

def potegowanie2(n, k):
    if k == 0:
        return 1
    if k == 1:
        return n
    if k > 1 and k % 2 == 0:
        wynik = potegowanie2(n, k / 2)
        return wynik * wynik
    if k > 1 and k % 2 == 1:
        wynik = potegowanie2(n, k // 2)
        return wynik * wynik * n

print(potegowanie2(3,3))