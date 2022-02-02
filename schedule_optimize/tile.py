import tvm
from tvm import te
import pdb

m = 512
n = 1024
k = 64
A = te.placeholder((m, k), name='A')
B = te.placeholder((k, n), name='B')
K = te.reduce_axis((0, k), name='K')
C = te.compute((m, n), lambda i, j: te.sum(A[i, K] * B[K, j], axis=K), name='C')

s = te.create_schedule(C.op)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")
# pdb.set_trace()
# xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 16, 64)

print(tvm.lower(s, [A, B, C], simple_mode=True))
