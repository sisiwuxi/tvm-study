import tvm
from tvm import relay,te

def test_conv():
  A1 = tvm.te.placeholder((1,2,30,30,64), name='input')
  W1 = tvm.te.placeholder((2,2,3,3,64,64), name='weight')

  rco1 = tvm.te.reduce_axis((0, 2), name='rco1')
  ry1 = tvm.te.reduce_axis((0, 3), name='ry1')
  rx1 = tvm.te.reduce_axis((0, 3), name='rx1')
  rci1 = tvm.te.reduce_axis((0, 64), name='rci1')
  stride_height = 1
  stride_width = 1

  B1 = tvm.te.compute((1,2,28,28, 64), lambda nn,ff,yy, xx, vlen1: tvm.te.sum(W1[ff,rco1,ry1,rx1,rci1,vlen1] * A1[nn, rco1, ry1 + stride_height*yy, rx1 + stride_width*xx,rci1], axis=[rco1,ry1, rx1, rci1]), name='output')

  s = tvm.te.create_schedule(B1.op)
  n,ko,h,w,ki  = s[B1].op.axis
  rco,ry,rx, rci = s[B1].op.reduce_axis

  w_factor_inner = 28
  tile_c = 1
  tile_h = 2
  wo, wi = s[B1].split(w, w_factor_inner)
  ho, hi = s[B1].split(h, tile_h)
  rco_o, rco_i = s[B1].split(rco, tile_c)
  s[B1].reorder(n,ko,rco_o,wo,ho,hi,rco_i,ry,rx,wi,ki,rci)
 
  #print(tvm.lower(s, [W1, A1, B1], simple_mode=True))

  def intrin():
    A = tvm.te.placeholder((1,3,3,64,64), name='w')
    B = tvm.te.placeholder((1,3,30,64), name='b')
    k = tvm.te.reduce_axis((0, 64), name='k')
    k_outer = tvm.te.reduce_axis((0, 1), name='k_outer')
    ry = tvm.te.reduce_axis((0, 3), name='ry')
    rx = tvm.te.reduce_axis((0, 3), name='rx')
    stride_width = 1
    C = tvm.te.compute((28,64),lambda m,n: tvm.te.sum(A[k_outer,ry,rx,k,n] * B[k_outer,ry, rx + m*stride_width,k], axis=[k_outer,ry,rx,k]),name='out')
    s1 = tvm.te.create_schedule(C.op)
    w,ofm  = s1[C].op.axis
    kco,ky,kx,kci = s1[C].op.reduce_axis
    s1[C].reorder(kco,ky,kx,w,ofm,kci)
    xx_ptr = tvm.tir.decl_buffer(A.shape, A.dtype, name="W",offset_factor=1, data_alignment=64)

    # yy_ptr = tvm.tir.decl_buffer(B.shape, B.dtype, name="some", offset_factor=1,strides=[30*30*64, 30*64, 64, 1],data_alignment=64)
    yy_ptr = tvm.tir.decl_buffer(B.shape, B.dtype, name="some", offset_factor=1, strides=[3*30*64, 30*64, 64, 1],data_alignment=64)    

    zz_ptr = tvm.tir.decl_buffer(C.shape, C.dtype, name="OUT",offset_factor=1, data_alignment=64)
    import pdb;pdb.set_trace()

    def intrin_func(ins, outs):
      body = tvm.tir.call_extern("int32","dummy",ins[0].access_ptr("r"),ins[1].access_ptr("r"),outs[0].access_ptr("w"))
      return body, None, body 

    # with tvm.build_config(data_alignment=64):
    return tvm.te.decl_tensor_intrin(C.op, intrin_func, name="GEMM", binds={A: xx_ptr, B: yy_ptr, C: zz_ptr})

  tensorized = intrin()
  s[B1].tensorize(rco_i, tensorized)
  print(tvm.lower(s, [W1, A1, B1], simple_mode=True))

if __name__ == "__main__":
  test_conv()