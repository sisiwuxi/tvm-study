import tvm
import tvm.testing
from tvm import te
import numpy
import timeit
import pdb

n = 1024
A = te.placeholder((n,), name='A')
k = te.reduce_axis((0, n), name='k')

B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s1 = s[B].fuse(ko, ki)
# print(s1)
# pdb.set_trace()
print(tvm.lower(s, [A, B], simple_mode=True))
