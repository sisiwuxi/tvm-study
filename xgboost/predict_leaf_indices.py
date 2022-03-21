"""
Demo for obtaining leaf index
=============================
"""
import os
import numpy as np
import xgboost as xgb
from common import *


def log_reg(y_hat, y):
    p = 1.0/(1.0 + np.exp(-y_hat))
    # g = p - y.get_label()
    h = p*(1.0 - p)
    return h


# load data in do training
# CURRENT_DIR = os.path.dirname(__file__)
# CURRENT_DIR = os.path.abspath(__file__)
CURRENT_DIR = os.getcwd()
dtrain = xgb.DMatrix(os.path.join(CURRENT_DIR, './data/agaricus.txt.train'))
dtest = xgb.DMatrix(os.path.join(CURRENT_DIR, './data/agaricus.txt.test'))
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 3
bst = xgb.train(param, dtrain, num_round, watchlist)

save_all_tree(bst, bst.best_iteration, bst.best_ntree_limit)

print('start testing predict the leaf indices')
# predict using first 2 tree
leafindex = bst.predict(dtest, iteration_range=(0, 2), pred_leaf=True, strict_shape=True)
print(leafindex.shape)
# import pdb;pdb.set_trace()
# print(leafindex)
# predict all trees
leafindex = bst.predict(dtest, pred_leaf=True)
print(leafindex.shape)
# 
# leafindex = bst.predict(dtest, pred_leaf=True, output_margin=True, pred_contribs=True, approx_contribs=True)
# print(leafindex.shape)
# print(leafindex)