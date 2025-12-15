import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (10, 5)
BankData = pd.read_csv("bank.csv")

#shows the data types of each column  and statistics related to it
print("Shape:", BankData.shape)
print(BankData.dtypes)
pd.set_option('display.max_columns', None)
print(BankData.describe(include="all"))

#no missing values
print(BankData.isnull().sum())


# target distribution
print(BankData['deposit'].value_counts())
print(BankData['deposit'].value_counts(normalize=True))
BankData['deposit'].value_counts().plot(kind='bar')
plt.title("Target Distribution: Deposit")
plt.xlabel("Deposit")
plt.ylabel("Count")
plt.show()

#define data and target
#X = BankData.drop(columns=["deposit"])
#y = BankData["deposit"]

#print("X shape:", X.shape)
#print("y shape:", y.shape)
