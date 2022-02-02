import tvm
from tvm import te
import pdb

n = 1024
A = te.placeholder((n,), name='A')
k = te.reduce_axis((0, n), name='k')

B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
# print(tvm.lower(s, [A, B], simple_mode=False))
print("---------cutting line---------")

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)
# pdb.set_trace()
print(tvm.lower(s, [A, B], simple_mode=True))
# print(tvm.lower(s, [A, B], simple_mode=False))
