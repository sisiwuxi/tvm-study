from __future__ import absolute_import, print_function

import tvm
from tvm import te


def intrin_ews(ro,co,data_type):
    a = te.placeholder((ro,co), dtype=data_type, name="a")
    b = te.placeholder((ro,co), dtype=data_type, name="b")
    c = te.compute((ro,co), lambda i,j: a[i,j] + b[i,j], name="c")
    # import pdb;pdb.set_trace()
    # Preview a generic schedule
    preview = te.create_schedule(c.op)
    print(tvm.lower(preview, [a, b, c], simple_mode=True))

    # Define buffers
    # Offset factor --> optimize for vectorized buffering
    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[4,1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[4,1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[4,1])

    def intrin_func(ins, outs):
        # create IR builder
        ib = tvm.tir.ir_builder.create()
        aa, bb = ins
        cc = outs[0]
        ib.emit(
            tvm.tir.call_extern(
                "float32",
                "ews",
                cc.access_ptr("w"),
                aa.access_ptr("r"),
                bb.access_ptr("r"),
                ro,
                co,
                bb.strides[0],
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


rows = 2
cols = 2
data_type = "float32"
# Create an instance
intrinsic = intrin_ews(rows,cols,data_type)

ro = 4
co = 4
# Create a tensorizable schedule
A = te.placeholder((ro,co), dtype=data_type, name="A")
B = te.placeholder((ro,co), dtype=data_type, name="B")
C = te.compute((ro,co), lambda i,j: A[i,j] + B[i,j], name="C")
# Create a vanilla schedule
s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1],x_factor=2,y_factor=2)
print(tvm.lower(s, [A, B, C], simple_mode=True))
# Get a handle to the axis
# x, y = s[C].op.axis
# Tensorize!
s[C].tensorize(xi, intrinsic)
print(tvm.lower(s, [A, B, C], simple_mode=True))