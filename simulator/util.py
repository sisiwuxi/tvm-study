

import numpy as np
import pdb
MAX_DIFF_GAP = 1e-4

class Util():

    def print_array(arr):
        for i in range(len(arr)):
            print(arr[i])
        return

    def print_array_tile(arr, tile):
        M, N = arr.shape
        tM, tN = tile
        for m in range(0, M, tM):
            for n in range(0, N, tN):
                print('m, n: ', m, n)
                arr_tile = arr[m : m + tM, n : n + tN]
                print_array(arr_tile)
        return

    def check_result(self, res_std, res, string):
        # pdb.set_trace()
        # err = np.sum(np.abs(res - res_std))
        sub = np.abs(res - res_std)
        check = sub.flatten()
        if check.max() > MAX_DIFF_GAP:
            # np.set_printoptions(threshold=np.inf)
            print('>>>>>>>>>>>>fail', string, '<<<<<<<<<<<<<')
            print('res_std:')
            print(res_std)
            print('res:')
            print(res)
            res_diff = res - res_std
            print('diff:')
            print(res_diff)
        else:
            print('>>>>>>>>>>>> success', string, '<<<<<<<<<<<<<')
        return

    def __call__(self):
        return
