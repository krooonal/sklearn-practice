import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
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

from sklearn.model_selection import GridSearchCV
# Set the parameters by cross-validation
tuned_parameters = [
  {'n_neighbors': [1,2,3,4,5,6,7,8,9,10],
  'p': [1, 2]},
]

# Tune the parameters of KNeighborsClassifier() (tuned_parameters) using GridSearchCV.
# Use scoring='accuracy', cv=3.
# START EDIT
grid_search = None
# Call the fit method.

# END EDIT

print('Finished!')

print("Best parameters set found on development set:")
print()
print(grid_search.best_params_)

print("Grid scores on development set:")
print()
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


y_pred = grid_search.predict(X_test)

# To calculate the accuracy of the model
acc_gs=accuracy_score(y_test, y_pred)

# To get the confusion Matrix
cm_gs=confusion_matrix(y_test,y_pred)
print('Accuracy of tuned KNN:',acc_gs*100)

print('Confusion Matrix is')
print(cm_gs)