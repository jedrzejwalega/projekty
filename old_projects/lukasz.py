import sys

litery = sys.argv[1]

with open(litery) as literki:
    plik = literki.read()
print(plik)

def liczenie_a():
    litery1 = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "w", "y", "z"]
    for litera in litery1:
        counter = 0
        for cos in plik:
            for cos2 in cos:
                if cos2 == litera:
                    counter += 1
        print(litera + " wystepuje " + str(counter) + " razy.")
    
liczenie_a()
