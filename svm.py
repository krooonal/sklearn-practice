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

from sklearn.svm import SVC

# START EDIT
# Create a SVC classifier with kernel='linear',random_state=0,degree=5.
svc_classifier=SVC(kernel='linear',random_state=0,degree=5)

# Call fit method on the training dataset
svc_classifier.fit(X_train,y_train)

# Store predictions of X_test in y_pred
y_pred=svc_classifier.predict(X_test)
# END EDIT

# To calculate the accuracy of the model
acc_svc=accuracy_score(y_test, y_pred)

# To get the confusion Matrix
cm_svc=confusion_matrix(y_test,y_pred)
print('Accuracy of Support Vector Classifier:',acc_svc*100)

print('Confusion Matrix is')
print(cm_svc)

