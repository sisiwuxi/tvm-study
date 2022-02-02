import tvm
from tvm import te
import pdb

m = 512
n = 1024
A = te.placeholder((m, n), name='A')
B = te.compute((m, n), lambda i, j: A[i, j],name='B')

s = te.create_schedule(B.op)

xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s1 = s[B].reorder(xi, yo, xo, yi)
print(s1)
# pdb.set_trace()
print(tvm.lower(s, [A, B], simple_mode=True))
