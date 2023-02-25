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

from sklearn.neighbors import KNeighborsClassifier

# START EDIT
# Create a K Neighbors classifier with n_neighbors=2.
knn_classifier=KNeighborsClassifier(n_neighbors=2)

# Call fit method on the training dataset
knn_classifier.fit(X_train,y_train)

# Store predictions of X_test in y_pred
y_pred=knn_classifier.predict(X_test)
# END EDIT

# To calculate the accuracy of the model
acc_knn=accuracy_score(y_test, y_pred)

# To get the confusion Matrix
cm_knn=confusion_matrix(y_test,y_pred)
print('Accuracy of KNN:',acc_knn*100)

print('Confusion Matrix is')
print(cm_knn)

