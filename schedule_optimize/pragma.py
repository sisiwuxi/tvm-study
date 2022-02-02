import tvm
from tvm import te
import pdb

n = 1024
m = 1024
A = te.placeholder((n, m), name='A')
k = te.reduce_axis((0, n), name='k')
l = te.reduce_axis((0, m), name = 'l')

B = te.compute((n,), lambda i: te.sum(A[i, l], axis=l), name='B')

s = te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=4)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s1 = s[B].pragma(ki, "unroll")
# s1 = s[B].pragma(ko, "vectorize") # error
pdb.set_trace()
print(tvm.lower(s, [A, B], simple_mode=True))
