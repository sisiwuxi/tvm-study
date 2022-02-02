import tvm
from tvm import te
import pdb

m = 1024
A = te.placeholder((m,), name="A")
B = te.compute((m,), lambda i: A[i] + 1, name="B")
C = te.compute((m,), lambda i: B[i] * 2, name="C")

s = te.create_schedule(C.op)
print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")
s[B].compute_inline()
print(tvm.lower(s, [A, B, C], simple_mode=True))