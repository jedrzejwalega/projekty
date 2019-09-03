import numpy as np 


def all_combinations(n):
    values = []
    if np.isnan(n).any() == True:
        return "Error: function doesn't accept NaN."
    if type(n) not in [int, float]:
        return "Error: function only accepts integrers and floats"
    if n < 1:
        return "Error: n must be bigger or equal to 1"
    if n == 1:
        nums = ["0", "1"]
        return nums
    else:
        nums = all_combinations(n-1)
        for num in nums:
            new1 = list(num)
            new1.insert(0, "0")
            values.append(new1)
            new2 = list(num)
            new2.insert(0, "1")
            values.append(new2)
            values.sort()
        return values
        

print(all_combinations(3)) 

print(len(all_combinations(3)))


#TIP od Lukasza na przyszlosc:

# n = 3
# 000
# 001
# 010
# 011
# 100
# 101
# 110
# 111

# dla n = 4 jest 2 razy więcej, bo 0 albo 1 w nowej kolumnie może tylko być:

# 0000
# 0001
# 0010
# 0011
# 0100
# 0101
# 0110
# 0111

# 1000
# 1001
# 1010
# 1011
# 1100
# 1101
# 1110
# 1111


