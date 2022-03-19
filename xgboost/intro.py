import xgboost as xgb
import numpy as np


# numpy
data = np.random.rand(5, 10)  # 5 entities, each contains 10 features
label = np.random.randint(2, size=5)  # binary target
dtrain = xgb.DMatrix(data, label=label)
dtest = xgb.DMatrix(data, label=label)

# # scipy
# csr = scipy.sparse.csr_matrix((dat, (row, col)))
# dtrain = xgb.DMatrix(csr)

# # pandas
# data = pandas.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
# label = pandas.DataFrame(np.random.randint(2, size=4))
# dtrain = xgb.DMatrix(data, label=label)

# # saving
# dtrain = xgb.DMatrix('train.svm.txt')
# dtrain.save_binary('train.buffer')

# # missing value
# dtrain = xgb.DMatrix(data, label=label, missing=np.NaN)

# # weight
# w = np.random.rand(5, 1)
# dtrain = xgb.DMatrix(data, label=label, missing=np.NaN, weight=w)

# # load libsvm
# dtrain = xgb.DMatrix('train.svm.txt')
# dtest = xgb.DMatrix('test.svm.buffer')

# # load csv
# # label_column specifies the index of the column containing the true label
# dtrain = xgb.DMatrix('train.csv?format=csv&label_column=0')
# dtest = xgb.DMatrix('test.csv?format=csv&label_column=0')

# booster parameters
param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

# param['eval_metric'] = ['auc', 'ams@0']
# alternatively:
# plst = param.items()
# plst += [('eval_metric', 'ams@0')]

evallist = [(dtest, 'eval'), (dtrain, 'train')]

# train
num_round = 10
bst = xgb.train(param, dtrain, num_round, evallist)
# train(..., evals=evals, early_stopping_rounds=10)

# save
bst.save_model('0001.model')

# dump model
bst.dump_model('dump.raw.txt')
# # dump model with feature map
# bst.dump_model('dump.raw.txt', 'featmap.txt')

# bst = xgb.Booster({'nthread': 4})  # init model
# bst.load_model('model.bin')  # load data

import pdb;pdb.set_trace()
# 7 entities, each contains 10 features
data = np.random.rand(7, 10)
dtest = xgb.DMatrix(data)
ypred = bst.predict(dtest)

# ypred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))

# # plotting
# xgb.plot_importance(bst)
# xgb.plot_tree(bst, num_trees=2)
xgb.to_graphviz(bst, num_trees=2)

# # Use "gpu_hist" for training the model.
# reg = xgb.XGBRegressor(tree_method="gpu_hist")
# # Fit the model using predictor X and response y.
# reg.fit(X, y)
# # Save model into JSON format.
# reg.save_model("regressor.json")

# booster: xgb.Booster = reg.get_booster()