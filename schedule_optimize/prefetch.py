import tvm
from tvm import te
import pdb
n = 1024
dtype = "float32"
k = te.reduce_axis((0, n), name='k')
A = te.placeholder((n, n), dtype=dtype, name='A')
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s1 = s[B].prefetch(A, s[B].op.reduce_axis[0], 1)
# s1 = s[B].prefetch(A, s[B].op.reduce_axis[0], 3)
# pdb.set_trace()
print(tvm.lower(s, [A, B], simple_mode=True))
