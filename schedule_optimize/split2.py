import tvm
from tvm import te
import pdb

m = 1024
A = te.placeholder((m,), name='A')
B = te.compute((m,), lambda i: A[i], name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

bx, tx = s[B].split(B.op.axis[0], factor=64)
# bx, tx = s[B].split(B.op.axis[0], nparts=64)
# pdb.set_trace()
print(tvm.lower(s, [A, B], simple_mode=True))
