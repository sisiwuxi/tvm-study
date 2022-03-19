"""
Demo for using xgboost with sklearn
===================================
"""
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_california_housing
import xgboost as xgb
import multiprocessing
from common import *

if __name__ == "__main__":
    print("Parallel Parameter optimization")
    # X.shape=(20640, 8); y.shape=(20640,)
    # feature_names = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
    #                  "Population", "AveOccup", "Latitude", "Longitude"]
    # X[0]
    # array([   8.3252    ,   41.        ,    6.98412698,    1.02380952,
    #            322.        ,    2.55555556,   37.88      , -122.23      ])   
    X, y = fetch_california_housing(return_X_y=True)
    xgb_model = xgb.XGBRegressor(n_jobs=multiprocessing.cpu_count() // 2)
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 4, 6],
                                   'n_estimators': [50, 100, 200]}, verbose=1,
                                    n_jobs=2)
    clf_fit = clf.fit(X, y)
    # use two feature 6 & 7 only
    # clf_fit = clf.fit(X[:,[6,7], y)
    print(clf.best_score_)
    print(clf.best_params_)
    print("best_index_=%d,best_score_=%f"%(clf.best_index_, clf.best_score_))
    bst = clf.best_estimator_
    save_all_tree(bst, clf.best_index_, clf.best_params_['n_estimators'])

    # bst = xgb_model.fit(X,y)
    # # use two feature 6 & 7 only
    # # bst = xgb_model.fit(X[:,[6,7]],y)
    # bst.save_model('0001.model')
    # print("bst.n_estimators=",bst.n_estimators)
    # # for i in range(bst.n_estimators):
    # for i in range(10):
    #     dot_data = xgb.to_graphviz(bst, num_trees=i)
    #     dot_path = "./dot/" + str(i) + ".dot"
    #     png_path = "./dot/" + str(i) + ".png"
    #     dot_data.save(dot_path)
    #     cmd = "dot -Tpng " + dot_path + " -o " + png_path
    #     os.system(cmd)