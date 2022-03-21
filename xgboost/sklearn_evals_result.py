"""
Demo for accessing the xgboost eval metrics by using sklearn interface
======================================================================
"""

import xgboost as xgb
import numpy as np
from sklearn.datasets import make_hastie_10_2
from common import *
# X.shape=(2000, 10), y.shape=(2000,)
X, y = make_hastie_10_2(n_samples=2000, random_state=42)

# Map labels from {-1, 1} to {0, 1}
labels, y = np.unique(y, return_inverse=True)

X_train, X_test = X[:1600], X[1600:] # X_train.shape=(1600, 10); X_test.shape=(400, 10)
y_train, y_test = y[:1600], y[1600:] # y_train.shape=(1600,); p y_test.shape=(400,)

param_dist = {'objective':'binary:logistic', 'n_estimators':2}

clf = xgb.XGBModel(**param_dist)
# Or you can use: clf = xgb.XGBClassifier(**param_dist)

clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='logloss',
        verbose=True)
# import pdb;pdb.set_trace()
# Load evals result by calling the evals_result() function
evals_result = clf.evals_result()

print('Access logloss metric directly from validation_0:')
print(evals_result['validation_0']['logloss'])

print('')
print('Access metrics through a loop:')
for e_name, e_mtrs in evals_result.items():
    print('- {}'.format(e_name))
    for e_mtr_name, e_mtr_vals in e_mtrs.items():
        print('   - {}'.format(e_mtr_name))
        print('      - {}'.format(e_mtr_vals))

print('')
print('Access complete dict:')
print(evals_result)
bst = clf
save_all_tree(bst, bst.best_iteration, bst.best_ntree_limit)