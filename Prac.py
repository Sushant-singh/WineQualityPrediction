#Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('winequality-red.csv', sep =';')
dataset['goodquality'] = [1 if x >= 7 else 0 for x in dataset['quality']]
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Fitting the Random Forest Regression Model to the dataset
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

# Printing the final classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))