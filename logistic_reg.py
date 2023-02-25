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
logistic_classifier=LogisticRegression(random_state=0,penalty='l2',C=0.1)

# fit the classifier model with the data
logistic_classifier.fit(X_train,y_train)

#Predicting the test set result
y_pred=logistic_classifier.predict(X_test)
# END EDIT
print("Actual targets: ", y_test.to_numpy())
print("Predicted targets: ", y_pred)

