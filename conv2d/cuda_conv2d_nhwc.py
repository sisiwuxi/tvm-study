# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain LHS copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required bm applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
tvm/tutorials/optimize/opt_conv_cuda.py
"""
from sympy import factor
import tvm
from tvm import te
import os
from tvm.contrib import nvcc
import numpy as np
import tvm.testing
import pdb
from tvm.contrib import tedd

TASK = "conv2d"
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx")
    return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_conv2d.cu" % TASK).read()
    return code

def test_conv2d_nhwc_hwio():
    # conv2d
    N = 128
    Hi = 224
    Wi = 224
    Ci = 3
    R = 7
    S = 7
    Co = 64
    Ho = 112
    Wo = 112

    stride = [2,2]
    padding = [3,3,3,3]
    dialation = [1,1]

    Ho = (Hi+padding[0]+padding[1]-R)//stride[0]+1
    Wo = (Wi+padding[2]+padding[3]-S)//stride[1]+1

    # tile consts
    tile_n = 2
    tile_c = 2 #1
    num_thread_n = 8
    num_thread_c = 8
    vthread_n = 2
    vthread_c = 8
    step = 3
    vec_factor = 2 # 1,2,4,8,16

    block_factor_c = tile_c * num_thread_c * vthread_c # 2*8*8=128
    offset = 8
    A_align = step + offset # 3+8 = 11
    W_align = block_factor_c + offset # 128+8=136

    # Algorithm
    # layout == "NHWC":
    # kernel_layout == "HWIO"
    A = te.placeholder((N,Hi,Wi,Ci), name="A")
    W = te.placeholder((R,S,Ci,Co), name="W")

    # Pad input
    Apad = te.compute(
        (N,Hi+padding[0]+padding[1],Wi+padding[2]+padding[3],Ci),
        lambda nn, yy, xx, cc: tvm.tir.if_then_else(
            tvm.tir.all(yy>=padding[0], yy-padding[0]<Hi, xx>=padding[2], xx-padding[2]<Wi),
            A[nn, yy-padding[0], xx-padding[1], cc],
            tvm.tir.const(0.0, "float32"),
        ),
        name="Apad",
    )
    # Create reduction variables
    rc = te.reduce_axis((0, Ci), name="rc")
    ry = te.reduce_axis((0, R), name="ry")
    rx = te.reduce_axis((0, S), name="rx")
    # Compute the convolution
    B = te.compute(
        (N, Ho, Wo, Co),
        lambda nn, yy, xx, co: te.sum(
            Apad[nn, yy*stride[0]+ry, xx*stride[1]+rx, rc] * W[ry, rx, rc, co], axis=[ry, rx, rc]
        ),
        name="B",
    )
    s = te.create_schedule(B.op)
    # Memory Hierarchy
    # Designate the memory hierarchy
    s[Apad].compute_inline()  # compute Apad inline
    # s[W].compute_inline()
    BL = s.cache_write(B, "local")
    AA = s.cache_read(Apad, "shared", [BL])
    WW = s.cache_read(W, "shared", [BL])
    AL = s.cache_read(AA, "local", [BL])
    WL = s.cache_read(WW, "local", [BL])

    # Blocking
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis((0, num_thread_c), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread_n), "threadIdx.y")
    thread_xz = te.thread_axis((0, vthread_c), "vthread", name="vx")
    thread_yz = te.thread_axis((0, vthread_n), "vthread", name="vy")

    # schedule for output
    n,ho,wo,co = s[B].op.axis # n,ho,wo,co = 128,112,112,64
    # co
    co,vec = s[B].split(co, factor = vec_factor) # co,vec = 32,2
    s[B].vectorize(vec)
    tx,co = s[B].split(co, factor=tile_c) # tx,co = 16,2
    txz,tx = s[B].split(tx, factor=num_thread_c) # txz,tx = 2,8; num_thread_c=8
    bz,txz = s[B].split(txz, factor=vthread_c) # bz,txz = 1,2; vthread_c=8
    
    # n
    ty,n = s[B].split(n, factor=tile_n) # ty,n = 64,2
    tyz,ty = s[B].split(ty, factor=num_thread_n) # tyz,ty = 8,8; num_thread_n=8
    by,tyz = s[B].split(tyz, factor=vthread_n) # by,tyz = 2,4; vthread_n=4

    s[B].reorder(wo,by,bz,tyz,ho,txz,ty,tx,n,co,vec) # 112,4,1,2,112,2,8,8,2,2,2
    # s[B].reorder(wo,ho,by,bz,tyz,txz,ty,tx,n,co,vec) # 112,112,4,1,2,2,8,8,2,2,2
    
    # s[B].bind(bz, block_z) # 1 bz,txz,tx,co = 1,2,8,2
    # s[B].bind(c, block_y) # 4 by,tyz,ty,n = 4,2,8,2
    # s[B].bind(wo, block_x) # 112 wo
    s[B].bind(by, block_z) # 4 by,tyz,ty,n = 2,4,8,2
    s[B].bind(ho, block_y) # 112 wo
    s[B].bind(wo, block_x) # 112 wo

    s[B].bind(tyz, thread_yz) # 2,2 by,tyz,ty,n = 4,2,8,2; vthread_n=2
    s[B].bind(txz, thread_xz) # 2,8 bz,txz,tx,co = 1,2,8,2; vthread_c=8
    s[B].bind(ty, thread_y) # 8,8 by,tyz,ty,n = 4,2,8,2
    s[B].bind(tx, thread_x) # 8,8 bz,txz,tx,co = 1,2,8,2

    # # schedule local computation
    s[BL].compute_at(s[B], tx)
    n,ho,wo,co = s[BL].op.axis # n,ho,wo,co = 2,112,1,2
    rh,rw,rc = s[BL].op.reduce_axis # rh,rw,rc = 7,7,3
    rco,rci = s[BL].split(rc,factor=step) # rco,rci = 1,3
    s[BL].vectorize(co) # 4 ???
    s[BL].reorder(rco,rh,rw,rci,n,co) # 1,7,7,3,2,4
    s[AA].compute_at(s[BL], rw)
    s[WW].compute_at(s[BL], rw)
    s[AL].compute_at(s[BL], rci)
    s[WL].compute_at(s[BL], rci)

    # # schedule for data's share memory
    n,hi,wi,ci = s[AA].op.axis # n,hi,wi,ci=128,229,229,3
    s[AA].reorder(hi,wi,n,ci) # 229,229,128,3
    s[AA].storage_align(wi, A_align-1, A_align) # wi=229,A_align=11
    t = s[AA].fuse(n,ci) # t=n*ci=32*3=96
    ty,tx = s[AA].split(t, factor=num_thread_c) # num_thread_c=8, ty,tx = 12,8
    tz,ty = s[AA].split(ty, factor=num_thread_n) # num_thread_n=8, tz,ty = 12/8,8
    s[AA].bind(tx, thread_x) # num_thread_c=8, tx=8
    s[AA].bind(ty, thread_y) # num_thread_n=8, ty=8

    # schedule for kernel's share memory
    wr,ws,wci,wco = s[WW].op.axis # wr,ws,wci,wco=7,7,3,64
    # import pdb;pdb.set_trace()
    s[WW].storage_align(wci, W_align-1, W_align) # wci=3,W_align=136
    t = s[WW].fuse(wci,wco) # t=wci*wco=3*64=192
    t,vec = s[WW].split(t, factor=num_thread_c)# num_thread_c=8, t,vec = 24,8
    s[WW].vectorize(vec) # vec=8
    ty,tx = s[WW].split(t, factor=num_thread_c) # num_thread_c=8, ty,tx = 3,8
    tz,ty = s[WW].split(ty, factor=num_thread_n) # num_thread_n=8, tz,ty = 3/8,8
    s[WW].bind(tx, thread_x)
    s[WW].bind(ty, thread_y)


    mod = tvm.lower(s, [A, W, B], simple_mode=True, name="conv2d")
    print(mod.astext(show_meta_data=False))
    # # Tensor Expression Debug Display (TEDD)
    # tedd.viz_dataflow_graph(s, dot_file_path="/tmp/dfg.dot")
    # tedd.viz_schedule_tree(s, dot_file_path="/tmp/scheduletree.dot")
    # s = s.normalize()
    # tedd.viz_schedule_tree(s, dot_file_path="/tmp/scheduletree2.dot")
    # tedd.viz_itervar_relationship_graph(s, dot_file_path="/tmp/itervar.dot")
    return

def test_conv2d_hwcn_hwio():
    # The sizes of inputs and filters
    batch = 128
    in_channel = 3
    out_channel = 64
    in_size = 224
    kernel = 7
    pad = 3
    stride = 2

    # Algorithm
    A = te.placeholder((in_size, in_size, in_channel, batch), name="A")
    W = te.placeholder((kernel, kernel, in_channel, out_channel), name="W")
    out_size = (in_size - kernel + 2 * pad) // stride + 1
    # Pad input
    Apad = te.compute(
        (in_size + 2 * pad, in_size + 2 * pad, in_channel, batch),
        lambda yy, xx, cc, nn: tvm.tir.if_then_else(
            tvm.tir.all(yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size),
            A[yy - pad, xx - pad, cc, nn],
            tvm.tir.const(0.0, "float32"),
        ),
        name="Apad",
    )
    # Create reduction variables
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel), name="ry")
    rx = te.reduce_axis((0, kernel), name="rx")
    # Compute the convolution
    B = te.compute(
        (out_size, out_size, out_channel, batch),
        lambda yy, xx, ff, nn: te.sum(
            Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff], axis=[ry, rx, rc]
        ),
        name="B",
    )

    s = te.create_schedule(B.op)
    mod = tvm.lower(s, [A, W, B], simple_mode=True, name="conv2d")
    print(mod.astext(show_meta_data=False))
    return

if __name__ == "__main__":
    # test_conv2d_nchw_oihw()
    # test_conv2d_hwcn_hwio()
    test_conv2d_nhwc_hwio()
    # test_conv2d_hwnc_hwoi()
    # test_conv2d_nchw4c_oihw4o4i()