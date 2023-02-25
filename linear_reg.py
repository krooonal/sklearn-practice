import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
data = load_diabetes(as_frame=True).frame
print(data.head())
X = data.drop("target", axis=1)
y = data["target"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression

# START EDIT
# Create a linear regressor.
regressor_lr =LinearRegression()

# Call fit method on the training dataset
regressor_lr.fit(X_train,y_train)

# Store predictions of X_test in y_pred
y_pred=regressor_lr.predict(X_test)
# END EDIT

from sklearn.metrics import mean_squared_error
err_lr=mean_squared_error(y_test, y_pred, squared=False)
print('Root mean squared error:', err_lr)