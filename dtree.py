import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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

from sklearn.tree import DecisionTreeClassifier

# START EDIT
# Create a Decision Tree classifier with criterion='entropy',random_state=0.
classifier_decision_tree=None

# Call fit method on the training dataset

# Store predictions of X_test in y_pred
y_pred=[]
# END EDIT

# To calculate the accuracy of the model
acc_decision=accuracy_score(y_test, y_pred)

# To get the confusion Matrix
cm_decision=confusion_matrix(y_test,y_pred)
print('Accuracy of Decision Tree:',acc_decision*100)

print('Confusion Matrix is')
print(cm_decision)

