import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
data = load_wine(as_frame=True).frame
print(data.head())
X = data.drop("target", axis=1)
y = data["target"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression

# START EDIT
# Create a logistic regression classifier.
# Use random_state=0,penalty='l2',C=0.1
logistic_classifier=None

# Call fit method on the training dataset

# Store predictions of X_test in y_pred
y_pred=[]
# END EDIT

print("Actual targets: ", y_test.to_numpy())
print("Predicted targets: ", y_pred)

