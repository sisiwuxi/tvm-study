import tvm
from tvm import te
import pdb

n = te.var("n")
m = te.var("m")

A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")

s = te.create_schedule(C.op)
s[B].compute_at(s[C], C.op.axis[0])
print(tvm.lower(s, [A, B, C], simple_mode=True))

print("---------cutting line---------")
s[B].compute_at(s[C], C.op.axis[0])
s1 = s[B].compute_root()
# s[C].compute_root()
# pdb.set_trace()
print(tvm.lower(s, [A, B, C], simple_mode=True))

