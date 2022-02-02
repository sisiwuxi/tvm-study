import tvm
from tvm import te
import pdb
n = 1024
k = te.reduce_axis((0, n), name='k')

A = te.placeholder((n,), name='A')
B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)
ko, ki = s[B].split(s[B].op.reduce_axis[0], 32)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

# BR = s.rfactor(B, ki)
BR = s.rfactor(B, ko)
# pdb.set_trace()
print(tvm.lower(s, [A, B], simple_mode=True))
