import numpy as np

def exercise_2(word: str):
    if type(word) is int:
        return "Error: function doesn't accept integrers"
    if type(word) == float:
        return "Error: function doesn't accept floats or NaN"
    if len(word) == 1:
        return "Error: single letter is not a word"
    palindrome_candidate = ""
    for index in range(len(word)):
        if index == 0:
            continue
        palindrome_candidate += word[-index]
    palindrome_candidate += word[0]
    if palindrome_candidate == word:
        return "Yes"
    else:
        return "No"
