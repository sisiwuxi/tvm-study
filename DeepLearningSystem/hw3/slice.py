import numpy as np

A = np.random.randn(4,5,6)
import pdb;pdb.set_trace()
print(A[0:4:1])
print(A[0:4:1, 2:5:1, 3:4:1])
print(A[0::2, 2:5:1, ::2])