import tvm
from tvm import te
from tvm import relay
import pdb

N, M, L = 1024, 512, 64
A = te.placeholder((N, L), name='A')
B = te.placeholder((M, L), name='B')
k = te.reduce_axis((0, L), name='k')
# declare behavior
C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k), name='C')
s = te.create_schedule(C.op)

# lowering rule to generate hardware intrinsics to carry out the computation
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
        # arm
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




# w,x = te.placeholder((8,8), te.placeholder(8,8))
# k = te.reduce_axis((0,8))
# # declare behavior
# y = te.compute((8,8), lambda i,j: te.sum(w[i,k]*x[j,k], axis=k))

# # lowering rule to generate hardware intrinsics to carry out the computation
# def gemm_intrin_lower(inputs, outputs):
#     ww_ptr = inputs[0].access_ptr("r")
#     xx_ptr = inputs[1].access_ptr("r")
#     zz_ptr = outputs[0].access_ptr("w")
#     compute = te.hardware_intrin("gemm8x8", ww_ptr, xx_ptr, zz_ptr)
#     reset = te.hardware_intrin("fill_zero", zz_ptr)
#     update = te.hardware_intrin("fuse_gemm8x8_add", ww_ptr, xx_ptr, zz_ptr)
#     return compute, reset, update

# gemm8x8 = t.decl_tensor_intrin(y.op, gemm_intrin_lower)

# # --------------------------------------
# out_buffer CL[8][8]
# in_buffer AL[8][8], BL[8][8]

# for each yo in 0..128:
#     for each xo in 0..128:
#         acc.fill_zero(CL)
#         for each ko in 0..128:
#             # compiler dectect n-dimensional copy pattern and generate copy intrinsic
#             acc.dma_copy2d(AL, A[yo*8 : yo*8 + 8][ko])
#             acc.dma_copy2d(BL, B[xo*8 : xo*8 + 8][ko])
#             # tensor compute intrinsic
#             acc.fuse_gemm8x8_add(AL, BL, CL)
#         acc.dma_copy2d(C[yo*8: yp*8+8][xo*8 : xo*8+8], CL)