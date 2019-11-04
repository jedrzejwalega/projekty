# ##STINGS

# #Creating a string

# s = ' Hello '
# t = "Hello"
# m = """This is a long string that is
# spread across two lines."""

# ## input What is a difference?

# num = eval(input( ' Enter a number: ' ))
# string = input( ' Enter a string: ' )

# #the string length

# len( ' Hello ' )

# ##Concatenation and repetitions

# ' AB ' + ' cd '
# ' A ' + ' 7 ' + ' B '
# ' Hi ' *4

# #Further example

# s = ''
# for i in range(10):
#     t = input( ' Enter a letter: ' )
   
#         s = s + t
# print(s)
# #The in operator
# string=input('podaj ciag znakow :')
# if 'a' in string:
#     print( ' Your string contains the letter a. ' )
# else:
#     print("no to dupa")

# if ';' not in string:
# print( ' Your string does not contain any semicolons. ' )

# ## abbreviation
#  if t=='a' or t=='e' or t=='i' or t=='o' or t=='u' :
#  if t in 'aeiou' :

# ##
# s='pawel'
#  #INDEXING
#     s[0] --- p
#     s[1] --- a
#     s[-1]--- l
#     s[-2] --- e
# #SLICES

# s[2:5] ---- elementy 2,3,4
# s[:5] --- elementy 0,1,2,3,4
# s[5:] --- od elementu 5 do konca
# s[-2:] --- od przedostatniego do konca
# s[1:7:2] --- od 1 do do 7 co 2 1,3,5
# s[::-1] --- w odwrotnej kolejnosci (wypisz w wspak)

# #Changing individual character of a string

# s = s[:4] + ' X ' + s[5:]

# ##LOOPING

# for i in range(len(name)):
#     print (name[i])
# #OR

# for c in name:
#     print(c)

# ##STRING METHODS

# name='PAWEL'
# name.lower()
# name.upper()
#lower() ---- returns a string with every letter of the original in lowercase
#upper()---- returns a string with every letter of the original in uppercase
#replace(x,y)--- returns a string with every occurrence of x replaced by y
#count(x)--- counts the number of occurrences of x in the string
#index(x)--- returns the location of the first occurrence of x
#isalpha()--- returns True if every character of the string is a letter

##EXERCISES

#1. Write a program that asks the user to enter a string. The program should then print the
#following:
#   (a) The total number of characters in the string
#   (b) The string repeated 10 times
#   (c) The first character of the string (remember that string indices start at 0)
#   (d) The first three characters of the string
#   (e) The last three characters of the string
#   (f) The string backwards
#   (g) The seventh character of the string if the string is long enough and a message otherwise
#   (h) The string with its first and last characters removed
#   (i) The string in all caps
#   (j) The string with every a replaced with an CHAPTER 
#   (k) The string with every letter replaced by a space

def number_one():
    user_inp = str(input("Please enter the string, doesn't have to be in ticks: "))
    inp_len = len(user_inp)
    return inp_len
print(number_one())

#2. A simple way to estimate the number of words in a string is to count the number of spaces
#in the string. Write a program that asks the user for a string and returns an estimate of how
#many words are in the string.
#3. People often forget closing parentheses when entering formulas. Write a program that asks
#the user to enter a formula and prints out whether the formula has the same number of open-
#ing and closing parentheses.
#4. Write a program that asks the user to enter a word and prints out whether that word contains
#any vowels.
#5. Write a program that asks the user to enter a string. The program should create a new string
#called new_string from the user’s string such that the second character is changed to an
#asterisk and three exclamation points are attached to the end of the string. Finally, print
#new_string . Typical output is shown below:
#Enter your string: Qbert
#Q*ert!!!
#6. Write a program that asks the user to enter a string s and then converts s to lowercase, re-
#moves all the periods and commas from s , and prints the resulting string.
#7. Write a program that asks the user to enter a word and determines whether the word is a
#palindrome or not. A palindrome is a word that reads the same backwards as forwards.
#8. At a certain school, student email addresses end with @student.college.edu , while pro-
#fessor email addresses end with @prof.college.edu . Write a program that first asks the
#user how many email addresses they will be entering, and then has the user enter those ad-
#dresses. After all the email addresses are entered, the program should print out a message
#indicating either that all the addresses are student addresses or that there were some profes-
#sor addresses entered.
#9. Ask the user for a number and then print the following, where the pattern ends at the number
#that the user enters.
#1
# 2
#  3
#   4
#10. Write a program that asks the user to enter a string, then prints out each letter of the string
#doubled and on a separate line. For instance, if the user entered HEY , the output would be
#HH
#EE
#YY
