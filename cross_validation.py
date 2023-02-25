import numpy as np
X = np.array(range(11))
y = np.array(["a", "a", "a", "b","b", "c", "c", "c", "c", "c", "c"])

from sklearn.model_selection import KFold

# we can define the number of splits
k_fold = KFold(n_splits=2, shuffle=False)

for train_indices, test_indices in k_fold.split(X, y):
  print('train y: %s | test y: %s' % (y[train_indices], y[test_indices]))

from sklearn.model_selection import StratifiedKFold

# we can define the number of splits
stratified_k_fold = StratifiedKFold(n_splits=2, shuffle=True)

for train_indices, test_indices in stratified_k_fold.split(X,y):
  print('train y: %s | test y: %s' % (y[train_indices],y[test_indices]))