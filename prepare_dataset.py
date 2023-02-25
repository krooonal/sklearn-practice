import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

data = load_wine(as_frame=True).frame
X = data.drop("target", axis=1)
y = data["target"]
print(X.head())
print(y.head())

# Split the data into train and test sets using sklearn train_test_split.
# Use test_size = 0.25 and random_state = 0.
from sklearn.model_selection import train_test_split

# START EDIT
X_train,X_test,y_train,y_test=([],[],[],[])
# END EDIT

# Scale the data using sklearn standard scaler
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()

# START EDIT
X_train=[]
X_test=[]
# END EDIT