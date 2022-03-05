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
from sympy import factor
import tvm
from tvm import te
import os
from tvm.contrib import nvcc
import numpy as np
import tvm.testing
import pdb
from tvm.contrib import tedd

TASK = "dense"
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
        code = open("perf/%s_manual.cu" % TASK).read()
    return code

def test_dense():
    # dense
    M = 128
    N = 1024
    K = 2048

    # Algorithm
    k = te.reduce_axis((0, K), name="k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((N, K), name="B")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[j, k], axis=k), name="C")
    
    # schedule
    s = te.create_schedule(C.op)

    # explicit memory access function
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "local", [C])
    BF = s.cache_read(BS, "local", [C])
    CF = s.cache_write(C, "local")
    CS = s.cache_read(CF, "shared", [C])

    # define the stride of intrinsic function
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # schedule for dense computation
    m, n = C.op.axis # m,n = 128, 1024
    block_i, mc = s[C].split(m, factor=16) # block_i, mc = 8, 16
    block_j, nc = s[C].split(n, factor=16) # block_j, nc = 64, 16
    s[C].reorder(block_i, block_j, mc, nc) # 8 / 64 / 16 / 16
    t = s[C].fuse(mc, nc) # t = 256
    t, vi = s[C].split(t, factor=1) # t, vi = 256, 1
    t, tx = s[C].split(t, factor=64) # t, tx = 4, 64
    t, ty = s[C].split(t, factor=1) # t, ty = 4, 1
    t, tz = s[C].split(t, factor=1) # t, tz = 4, 1
    s[C].bind(block_i, block_x) # block_i = 8 = block_x."thread_extent" = 8
    s[C].bind(block_j, block_y) # block_j = 64 = block_y."thread_extent" = 64
    s[C].bind(tz, thread_z) # tz = 1 = thread_z."thread_extent"
    s[C].bind(ty, thread_y) # ty = 1 = thread_y."thread_extent"
    s[C].bind(tx, thread_x) # tx = 64 = thread_x."thread_extent"
    s[C].vectorize(vi)

    # schedule for wmma store
    s[CS].compute_at(s[C], block_j) # block_j = 64
    sm, sn = CS.op.axis # sm,sn = 16,16
    s[CS].storage_align(sm, 16-1, 16) # sm = 16
    sm, smi = s[CS].split(sm, factor=16) # sm, smi = 1, 16
    sn, sni = s[CS].split(sn, factor=16) # sn, sni = 1, 16
    sm, smii = s[CS].split(sm, factor=1) # sm, smii = 1, 1
    sn, snii = s[CS].split(sn, factor=1) # sn, snii = 1, 1
    s[CS].reorder(sm, sn, smii, snii, smi, sni)
    s[CS].bind(sm, thread_y) # sm = 1, thread_y."thread_extent" = 1
    s[CS].bind(sn, thread_z) # sn = 1, thread_z."thread_extent" = 1

    # schedule for wmma computation
    s[CF].compute_at(s[CS], sn) # sn = 1, s[CS].bind(sn, thread_z)
    warp_i, warp_j = CF.op.axis # warp_i, warp_j = 16,16
    warp_i, _ii = s[CF].split(warp_i, factor=16) # warp_i, _ii = 1, 16
    warp_j, _jj = s[CF].split(warp_j, factor=16) # warp_j, _jj = 1, 16
    (k,) = CF.op.reduce_axis # k = 2048
    k, _k = s[CF].split(k, factor=16) # k, _k = 128, 16
    ko, ki = s[CF].split(k, factor=1) # ko, ki = 128, 1
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k) # 128 / 1 / 1 / 1 / 16 / 16 / 16

    # schedule for wmma_matrix_a load
    s[AF].compute_at(s[CF], ki) # ko, ki = 128, 1
    lm, lk = AF.op.axis # lm, lk = 16, 2048
    lm, lm_ii = s[AF].split(lm, factor = 16) # lm, lm_ii = 1, 16
    lk, lk_jj = s[AF].split(lk, factor = 16) # lk, lk_jj = 128, 16
    s[AF].reorder(lm, lk, lm_ii, lk_jj) # 1 / 128 / 16 / 16

    # schedule for wmma_matrix_b load
    s[BF].compute_at(s[CF], ki) # ko, ki = 128, 1
    ln, lk = s[BF].op.axis # ln, lk = 16, 2048
    ln, ln_ii = s[BF].split(ln, factor=16) # ln, ln_ii = 1, 16
    lk, lk_jj = s[BF].split(lk, factor=16) # lk, lk_jj = 128, 16
    s[BF].reorder(ln, lk, ln_ii, lk_jj)

    # schedule for A's shared memory load
    s[AS].compute_at(s[CF], ko) # ko, ki = 128, 1
    sm, sk = s[AS].op.axis # sm, sk = 16, 2048
    s[AS].storage_align(sm, 16-1, 16)
    t = s[AS].fuse(sm, sk) # t = 32768
    t, vi = s[AS].split(t, factor=1) # t, vi = 32768, 1
    t, tx = s[AS].split(t, factor=64) # t, tx = 512, 64
    t, ty = s[AS].split(t, factor=1) # t, ty = 512, 1
    t, tz = s[AS].split(t, factor=1) # t, tz = 512, 1 
    s[AS].bind(tz, thread_z) # tz = 1, thread_z."thread_extent" = 1
    s[AS].bind(ty, thread_y) # ty = 1, thread_y."thread_extent" = 1
    s[AS].bind(tx, thread_x) # tx = 64, thread_x."thread_extent" = 1
    s[AS].vectorize(vi)
    # pdb.set_trace()
    # schedule for B's shared memory load
    s[BS].compute_at(s[CF], ko) # ko, ki = 128, 1
    sn, sk = s[BS].op.axis # sn, sk = 16, 16
    s[BS].storage_align(sn, 16-1, 16)
    t = s[BS].fuse(sn, sk) # t = 256
    t, vi = s[BS].split(t, factor=1) # t, vi = 256, 1
    t, tx = s[BS].split(t, factor=64) # t, tx = 4, 64
    t, ty = s[BS].split(t, factor=1) # t, ty = 4, 1
    t, tz = s[BS].split(t, factor=1) # t, tz = 4, 1 
    s[BS].bind(tz, thread_z) # tz = 1, thread_z."thread_extent" = 1
    s[BS].bind(ty, thread_y) # ty = 1, thread_y."thread_extent" = 1
    s[BS].bind(tx, thread_x) # tx = 64, thread_x."thread_extent" = 1
    s[BS].vectorize(vi)

    # print(tvm.lower(s, [A, B, C], simple_mode=True))
    mod = tvm.lower(s, [A, B, C], simple_mode=True, name="dense")
    print(mod.astext(show_meta_data=False))
    # # Tensor Expression Debug Display (TEDD)
    tedd.viz_dataflow_graph(s, dot_file_path="/tmp/dfg.dot")
    tedd.viz_schedule_tree(s, dot_file_path="/tmp/scheduletree.dot")
    s = s.normalize()
    tedd.viz_schedule_tree(s, dot_file_path="/tmp/scheduletree2.dot")
    tedd.viz_itervar_relationship_graph(s, dot_file_path="/tmp/itervar.dot")



if __name__ == "__main__":
    test_dense()