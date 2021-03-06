import tvm
from tvm import te
import pdb
n = 1024
A = te.placeholder((n,), name='A')
k = te.reduce_axis((0, n), 'k')
B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
tx = te.thread_axis("threadIdx.x")
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s1 = s[B].set_store_predicate(tx.var.equal(0))
pdb.set_trace()
print(tvm.lower(s, [A, B], simple_mode=True))
