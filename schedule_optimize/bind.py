from tvm import te
import tvm
import pdb

n = 1024
A = te.placeholder((n,), name='A')
k = te.reduce_axis((0, n), name='k')

B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

b1 = s[B].bind(ko, te.thread_axis("blockIdx.x"))
b2 = s[B].bind(ki, te.thread_axis("threadIdx.x"))
# pdb.set_trace()
print(tvm.lower(s, [A, B], simple_mode=True))
