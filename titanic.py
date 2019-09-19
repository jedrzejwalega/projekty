import numpy as np
import pandas as pd

from sklearn import preprocessing
import matplotlib.pyplot as plt 

train_titanic = pd.read_csv("train.csv")
# print(train_titanic.head())

test_titanic = pd.read_csv("test.csv")
# print(test_titanic.head())

raw_train = train_titanic.copy()

raw_test = test_titanic.copy()


# number_samples_train = len(train_titanic.index)
# number_samples_test = len(test_titanic.index)

# print("test:", number_samples_test, "train:", number_samples_train)

train_nulls = train_titanic.isnull().sum()
test_nulls = test_titanic.isnull().sum()

# print(train_nulls, test_nulls)

#First, let's do test_titanic, to counter the tutorial:
# Age histogram with plot:
test_titanic["Age"].hist(density=True, color="blue", alpha=0.6)
test_titanic["Age"].plot(kind="density", color="blue")
plt.legend(["Age"])
plt.show()

# Replacing missing Age with median:
median_age = test_titanic["Age"].median()
print(median_age)

test_titanic["Age"].fillna(median_age, inplace=True)
test_nulls = test_titanic.isnull().sum()
print(test_nulls)

# Counting Embarked for each value type:
count_embarked = test_titanic["Embarked"].value_counts()
print(count_embarked)

# Plot for Embarked:
test_titanic["Embarked"].value_counts().plot(kind="bar")
plt.show()

# Replacing missing Embarked with S:
fill_embarked = test_titanic["Embarked"].value_counts().idxmax()
print(fill_embarked)
test_titanic["Embarked"].fillna(fill_embarked, inplace=True)
print(test_titanic["Embarked"].isnull().any())

# Dropping Cabin:
print(test_titanic.head())
test_titanic.drop("Cabin", axis=1, inplace=True)
print(test_titanic.head())

# Now let's readjust train_titanic, like in the tutorial:
# Replacing missing Age with median:
median_age = train_titanic["Age"].median()
print(median_age)
train_titanic["Age"].fillna(median_age, inplace=True)

# Replacing missing Embarked with S:
fill_embarked = train_titanic["Embarked"].value_counts().idxmax()
print(fill_embarked)
train_titanic["Embarked"].fillna(fill_embarked, inplace=True)
print(train_titanic["Embarked"].isnull().any())

# Dropping Cabin:
print(train_titanic.head())
train_titanic.drop("Cabin", axis=1, inplace=True)
print(train_titanic.head())

# Plot comparing Raw and Adjusted Age for test_titanic:
fig, ax = plt.subplots()
raw_test["Age"].plot(kind="density", color="teal") 
raw_test["Age"].hist(density=True, color="teal", alpha=0.4)
test_titanic["Age"].plot(kind="density", color="orange")
test_titanic["Age"].hist(density=True, color="orange", alpha=0.5)
ax.legend(["Raw Age", "Adjusted Age"])
plt.show()

# Plot comparing Raw and Adjusted Age for train_titanic:
fig, ax = plt.subplots()
raw_train["Age"].plot(kind="density", color="teal") 
raw_train["Age"].hist(density=True, color="teal", alpha=0.4)
train_titanic["Age"].plot(kind="density", color="orange")
train_titanic["Age"].hist(density=True, color="orange", alpha=0.5)
ax.legend(["Raw Age", "Adjusted Age"])
plt.show()

# Additional changes
# First to test_titanic:
# Creating TravelAlone in place of SibSp and Parch, 0 means at least one of them had a value of 1, which means the person wasn't traveling alone. In other words TravelAlone = 0 means with family, TravelAlone = 1 means without family
test_titanic["TravelAlone"] = np.where(test_titanic["SibSp"] + test_titanic["Parch"] > 0, 0, 1)
test_titanic.drop("SibSp", axis=1, inplace=True)
test_titanic.drop("Parch", axis=1, inplace=True)

#Creating categorical data for Pclass, Embarked and Sex; 1 = True, 0 = False:
test_final = pd.get_dummies(test_titanic, columns=["Pclass","Embarked","Sex"])
test_final.drop("Ticket", axis=1, inplace=True)
test_final.drop("Name", axis=1, inplace=True)
test_final.drop("PassengerId", axis=1, inplace=True)
test_final.drop("Sex_female", axis=1, inplace=True)
print(test_final.head())

# Now to train_titanic:
# Creating TravelAlone in place of SibSp and Parch, 0 means at least one of them had a value of 1, which means the person wasn't traveling alone. In other words TravelAlone = 0 means with family, TravelAlone = 1 means without family
train_titanic["TravelAlone"] = np.where(train_titanic["SibSp"] + train_titanic["Parch"] > 0, 0, 1)
train_titanic.drop("SibSp", axis=1, inplace=True)
train_titanic.drop("Parch", axis=1, inplace=True)

# Creating categorical data for Pclass, Embarked and Sex; 1 = True, 0 = False:
train_final = pd.get_dummies(train_titanic, columns=["Pclass","Embarked","Sex"])
train_final.drop("Ticket", axis=1, inplace=True)
train_final.drop("Name", axis=1, inplace=True)
train_final.drop("PassengerId", axis=1, inplace=True)
train_final.drop("Sex_female", axis=1, inplace=True)
print(train_final.head())

# Exploration of Age for train:
train_final["Age"][train_final["Survived"] == 1].plot(kind="density", color="green")
train_final["Age"][train_final["Survived"] == 0].plot(kind="density", color="red")
plt.legend(["Survived", "Died"])
plt.xlabel("Age")
plt.show()

age_survived = train_final[["Age", "Survived"]]
by_age_survived = train_final[["Age", "Survived"]].groupby("Age", as_index=False).mean()
print(age_survived)
print(by_age_survived)

plt.bar(by_age_survived["Age"], height=by_age_survived["Survived"])
plt.ylabel("Survived")
plt.xlabel("Age")
plt.show()

# Creating IsMinor:
train_final["IsMinor"] = np.where(train_final["Age"] < 16, 1, 0)
# print(train_final)

# Exploration of Fare:
train_final["Fare"][train_final["Survived"] == 1].plot(kind="density", color="green")
train_final["Fare"][train_final["Survived"] == 0].plot(kind="density", color="red")
plt.legend(["Survived", "Died"])
plt.xlabel("Fare")
plt.xlim(-40, 250)
plt.show()

# Exploration of Class (work in progress):


plt.bar(train_titanic["Pclass"], height=train_titanic["Survived"])

plt.show()