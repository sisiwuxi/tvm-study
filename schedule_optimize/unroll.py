import tvm
from tvm import te
import pdb

n = 1024
A = te.placeholder((n, n), name='A')
B = te.placeholder((n, n), name='B')
C = te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')

s = te.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor=4)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s1 = s[C].unroll(xi)
pdb.set_trace()

print(tvm.lower(s, [A, B, C], simple_mode=True))
