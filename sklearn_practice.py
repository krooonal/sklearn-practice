
from sklearn.preprocessing import LabelEncoder

# Exercise 1: Convert the string feature into a numerical feature.
train_features = ["Apple", "Orange", "Potato", "Tomato", "Grape", "Apple", "Tomato", "Grape"]
test_features = ["Apple", "Orange", "Tomato"]
le = LabelEncoder()
transformed_train = le.fit_transform(train_features)
transformed_test = le.transform(test_features)

print(train_features)
print(test_features)
print(transformed_train)
print(transformed_test)

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
# Exercise 2: Split data into train and test
data = load_wine(as_frame=True).frame
print(data.head())
X = data.drop("target", axis=1)
y = data["target"]
print(X.head())
print(y.head())
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
print(X_train.shape)
print(y_train.shape)

# Exercise 3: Scale the data using standard scaler
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
print(X_train)

