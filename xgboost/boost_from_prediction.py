"""
Demo for boosting from prediction
=================================
"""
import os
import xgboost as xgb
from common import *

CURRENT_DIR = os.getcwd()
dtrain = xgb.DMatrix(os.path.join(CURRENT_DIR, './data/agaricus.txt.train'))
dtest = xgb.DMatrix(os.path.join(CURRENT_DIR, './data/agaricus.txt.test'))
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
###
# advanced: start from a initial base prediction
#
print('start running example to start from a initial prediction')
# specify parameters via map, definition are same as c++ version
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
# train xgboost for 1 round
bst = xgb.train(param, dtrain, 1, watchlist)
save_all_tree(bst, bst.best_iteration, bst.best_ntree_limit)
# import pdb;pdb.set_trace()
# Note: we need the margin value instead of transformed prediction in
# set_base_margin
# do predict with output_margin=True, will always give you margin values
# before logistic transformation
ptrain = bst.predict(dtrain, output_margin=True)
ptest = bst.predict(dtest, output_margin=True)
dtrain.set_base_margin(ptrain)
dtest.set_base_margin(ptest)

print('this is result of running from initial prediction')
bst = xgb.train(param, dtrain, 1, watchlist)
save_all_tree(bst, bst.best_iteration, bst.best_ntree_limit)
print("end")
