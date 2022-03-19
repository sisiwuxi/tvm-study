import os
import xgboost as xgb

def save_all_tree(bst, best_index, n_estimators):
    # import pdb;pdb.set_trace()
    bst.save_model('0001.model')
    
    dot_data = xgb.to_graphviz(bst, num_trees=best_index)
    dot_data.save("./best.dot")
    cmd = "dot -Tpng best.dot -o best.png"
    os.system(cmd)
    for i in range(n_estimators):
    # for i in range(10):
        dot_data = xgb.to_graphviz(bst, num_trees=i)
        dot_path = "./dot/" + str(i) + ".dot"
        png_path = "./dot/" + str(i) + ".png"
        dot_data.save(dot_path)
        cmd = "dot -Tpng " + dot_path + " -o " + png_path
        os.system(cmd)