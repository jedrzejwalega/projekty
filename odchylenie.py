import numpy

#ilosc linii w dokumencie, funkcja:

def line_counting():
  with open("linie") as linki:
    linie = linki.read()
    n = 0
    for linia in linie:
      n += 1
  return n


def main():

  #utworzenie listy z kazda linijka:
  with open("linie") as line:
      line2 = line.read()
      linijki = line2.split("\n")

  #dlugosc kazdej linijki:

  linijki_dlugosci = []
  for linijka in linijki:
      linijki_dlugosci.append(len(linijka))
      


  #dlugosc wszystkich linijek:

  dlugosc_linijek = 0
  for dlugosc_linijki in linijki_dlugosci:
    dlugosc_linijek += dlugosc_linijki


  #ilosc linii w dokumencie:
  line_number = line_counting()



  wartosc1 = float(dlugosc_linijek)
  wartosc2 = float(line_counting())

  #srednia dlugosc linii w dokumencie:
  medium_line_number = float(wartosc1) / float(wartosc2)



  #odchylenie standardowe dlugosci linii:

  suma_do_sigmy = 0
  for dlugosc_linijki in linijki_dlugosci:
      matma = float(dlugosc_linijki) - float(medium_line_number)
      matma_do_kwadratu = float(matma) ** 2.0
      suma_do_sigmy += float(matma_do_kwadratu)
  suma_przez_n = float(suma_do_sigmy) / float(line_number)
  odchylenie_standardowe = suma_przez_n ** 0.5

  print("Srednia dlugosc linii w dokumencie wynosi {medium_line_number}, a odchylenie standardowe {odchylenie_standardowe}.".format(medium_line_number = medium_line_number, odchylenie_standardowe = odchylenie_standardowe))

  #teraz odchylenie standardowe dlugosci slow:

  #lista z kazdym slowem:

  slowa = line2.split()


  #dlugosc kazdego slowa:

  dlugosc_slow = []
  for slowo in slowa:
      dlugosc_slow.append(float(len(slowo)))

  #suma dlugosci slow:

  suma_dlugosci_slow = 0
  for dlugosc_slowa in dlugosc_slow:
      suma_dlugosci_slow += float(dlugosc_slowa)


  #ilosc slow:
  n = 0
  for slowo in slowa:
      n += 1


  #srednia dlugosc slowa:
  srednia_dlugosc_slowa = float(suma_dlugosci_slow) / float(n)


  #odchylenie standardowe dlugosci slow:

  suma_do_sigmy2 = 0
  for dlugosc_slowa in dlugosc_slow:
      matma2 = float(dlugosc_slowa) - float(srednia_dlugosc_slowa)
      matma_do_kwadratu2 = float(matma2) ** 2.0
      suma_do_sigmy2 += float(matma_do_kwadratu2)
  suma_przez_n2 = float(suma_do_sigmy2) / float(n) 
  odchylenie_standardowe2 = float(suma_przez_n2) ** 0.5
  print("Srednia dlugosc slowa w dokumencie wynosi {srednia_dlugosc_slowa}, a odchylenie standardowe {odchylenie_standardowe2}.".format(srednia_dlugosc_slowa = srednia_dlugosc_slowa, odchylenie_standardowe2 = odchylenie_standardowe2))

  # dlugosc linii, 2 sposob:
  line_length_mean = numpy.mean(linijki_dlugosci)

  line_length_stdev = numpy.std(linijki_dlugosci)

  print("Druga metoda dlugosc linijki w dokumencie tez wynosi {line_length_mean}, a odchylenie standardowe {line_length_stdev} :D".format(line_length_mean = line_length_mean, line_length_stdev = line_length_stdev))

  # dlugosc slow, 2 sposob:
  word_length_mean = numpy.mean(dlugosc_slow)

  word_length_stdev = numpy.std(dlugosc_slow)

  print("Druga metoda dlugosc slowa w dokumencie tez wynosi {word_length_mean}, a odchylenie standardowe {word_length_stdev} :D".format(word_length_mean = word_length_mean, word_length_stdev = word_length_stdev))

main()