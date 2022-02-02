import tvm
from tvm import te
from tvm import relay
import pdb

N, M, L = 1024, 512, 64
A = te.placeholder((N, L), name='A')
B = te.placeholder((M, L), name='B')
k = te.reduce_axis((0, L), name='k')
C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k), name='C')
s = te.create_schedule(C.op)

def intrin_gemv(m, l):
    a = te.placeholder((l,), name='a')
    b = te.placeholder((m, l), name='b')
    k = te.reduce_axis((0, l), name='k')
    c =  te.compute((m,), lambda i: te.sum(a[k] * b[i, k], axis=k), name='c')
    Abuf = tvm.tir.decl_buffer(a.shape, a.dtype, name='A', offset_factor=1, strides=[1])
    Bbuf = tvm.tir.decl_buffer(b.shape, b.dtype, name='B', offset_factor=1, strides=[te.var("s1"), 1])
    Cbuf = tvm.tir.decl_buffer(c.shape, c.dtype, name='C', offset_factor=1, strides=[1])
    
    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(tvm.tir.call_extern("int32", "gemv_update", cc.access_ptr("w"), aa.access_ptr("r"), bb.access_ptr("r"), m, l, bb.strides[0]))
        return ib.get()
    #with tvm.build_config(offset_factor=1):
    with relay.build_config(opt_level=0):
        return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Abuf, b: Bbuf, c: Cbuf})

factor = 16
x, y = C.op.axis
z, = C.op.reduce_axis
# yo, yi = s[C].split(y, factor=factor)
# s[C].reorder(x, yo, yi, z)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], factor, factor)

gemv = intrin_gemv(factor, L)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s1 = s[C].tensorize(yi, gemv)
# s1 = s[C].tensorize(yo, gemv) # error
# pdb.set_trace()
print(tvm.lower(s, [A, B, C], simple_mode=True))
