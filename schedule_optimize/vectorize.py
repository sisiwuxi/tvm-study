import tvm
from tvm import te
import numpy
import timeit
import pdb

M = 1024
N = 1024
A = te.placeholder((M, N), name='A')
B = te.placeholder((M, N), name='B')
C = te.compute(
           (M, N),
           lambda x, y: A[x, y] + B[x, y],
           name='C')

s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

# s1 = s[C].vectorize(yi)
# s1 = s[C].vectorize(xi)
# s1 = s[C].vectorize(xo)
s1 = s[C].vectorize(yo)
# pdb.set_trace()
print(tvm.lower(s, [A, B, C], simple_mode=True))
