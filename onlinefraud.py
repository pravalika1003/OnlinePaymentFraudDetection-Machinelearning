import pandas as pd
import numpy as np
import pickle
data = pd.read_csv(r"C:\Users\prava\Downloads\archive\onlinefraud.csv")
print(data.head())
data.isnull().sum()
print(data.shape)
data=data.dropna()
print(data.shape)
print(data.type.value_counts())
type = data["type"].value_counts()
transactions = type.index
quantity = type.values

import plotly.express as px
figure = px.pie(data,
             values=quantity,
             names=transactions,hole = 0.5,
             title="Distribution of Transaction Type")
figure.show()
# Checking correlation
# Selecting only numeric columns
numeric_data = data.select_dtypes(include=['number'])

# Calculating correlation
correlation = numeric_data.corr()

# Printing correlation with the "isFraud" column
print(correlation["isFraud"].sort_values(ascending=False))
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2,
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=42)
onlinefraud = DecisionTreeClassifier()
onlinefraud.fit(xtrain, ytrain)
print(onlinefraud.score(xtest, ytest))
features = np.array([[2, 9000.60, 9000.60, 0.0]])
print(onlinefraud.predict(features))
pickle.dump(onlinefraud,open('dct_model.pkl','wb'))