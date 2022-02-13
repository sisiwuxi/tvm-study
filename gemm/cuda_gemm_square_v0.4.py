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
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Example code to do square matrix multiplication."""
import tvm
from tvm import te
import os
from tvm.contrib import nvcc
from tvm.contrib import spirv
import numpy as np
import tvm.testing
import pdb

TASK = "gemm"
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

def test_gemm():
    # dot
    M = 1024
    N = 1024
    K = 1024

    # Algorithm
    k = te.reduce_axis((0, K), name="k")
    LHS = te.placeholder((M, K), name="LHS")
    RHS = te.placeholder((K, N), name="RHS")
    OUT = te.compute((M, N), lambda i, j: te.sum(LHS[i, k] * RHS[k, j], axis=k), name="OUT")   
    
    # schedule
    s = te.create_schedule(OUT.op)

    LHS_S = s.cache_read(LHS, "shared", [OUT])
    RHS_S = s.cache_read(RHS, "shared", [OUT])
    LHS_L = s.cache_read(LHS_S, "local", [OUT])
    RHS_L = s.cache_read(RHS_S, "local", [OUT])
    OUT_L = s.cache_write(OUT, "local")

    scale = 8#32
    num_thread = 8#4
    block_factor = scale*num_thread

    m, n = OUT.op.axis

    by, ty = s[OUT].split(m, factor=block_factor)
    ty, mi = s[OUT].split(ty, factor=num_thread)

    bx, tx = s[OUT].split(n, factor=block_factor)
    tx, ni = s[OUT].split(tx, factor=num_thread)

    s[OUT].reorder(by, bx, ty, tx, mi, ni)
    s[OUT].bind(by, te.thread_axis("blockIdx.y"))
    s[OUT].bind(bx, te.thread_axis("blockIdx.x"))
    s[OUT].bind(ty, te.thread_axis("threadIdx.y"))
    s[OUT].bind(tx, te.thread_axis("threadIdx.x"))

    # # m = 1024
    # by, ty = s[OUT].split(m, factor=128)
    # # by = 1024/128=8, ty = 128
    # ty, mi = s[OUT].split(ty, factor=4)
    # # ty = 128/4=32, mi = 4
    # # n = 1024
    # bx, tx = s[OUT].split(n, factor=128)
    # # bx = 1024/128=8, tx = 128
    # tx, ni = s[OUT].split(tx, factor=4)
    # # tx = 128/4=32, ni = 4
    # s[OUT].reorder(by, bx, ty, tx, mi, ni)
    # s[OUT].bind(by, te.thread_axis("blockIdx.y"))
    # s[OUT].bind(bx, te.thread_axis("blockIdx.x"))
    # s[OUT].bind(ty, te.thread_axis("threadIdx.y"))
    # s[OUT].bind(tx, te.thread_axis("threadIdx.x"))

    # # m = 1024
    # m, mi = s[OUT].split(m, factor=4)
    # # m = 256 * 4, mi=register compute=4, innermost
    # m, ty = s[OUT].split(m, factor=32)
    # # m = 8 * 32 * 4, ty=#thread=32
    # by, vy = s[OUT].split(m, factor=2)
    # # m = 2 * 4 * 32 * 4, by=8, vy=virtual thread=2
    # n, ni = s[OUT].split(n, factor=4)
    # n, tx = s[OUT].split(n, factor=32)
    # bx, vx = s[OUT].split(n, factor=2)
    # s[OUT].reorder(by, bx, vy, vx, ty, tx, mi, ni)
    # s[OUT].bind(vy, te.thread_axis("vthread"))
    # s[OUT].bind(vx, te.thread_axis("vthread"))
    # s[OUT].bind(by, te.thread_axis("blockIdx.y"))
    # s[OUT].bind(bx, te.thread_axis("blockIdx.x"))
    # s[OUT].bind(ty, te.thread_axis("threadIdx.y"))
    # s[OUT].bind(tx, te.thread_axis("threadIdx.x"))

    s[OUT_L].compute_at(s[OUT], tx)

    # m, n = OUT_L.op.axis
    # k = OUT_L.op.reduce_axis[0]
    # ko, ki = s[OUT_L].split(k, factor=64)
    # s[OUT_L].reorder(ko, ki, m, n)
    # s[LHS_S].compute_at(s[OUT_L], ko)
    # s[RHS_S].compute_at(s[OUT_L], ko)
    # s[LHS_L].compute_at(s[OUT_L], ki)
    # s[RHS_L].compute_at(s[OUT_L], ki)
    s[LHS_S].compute_at(s[OUT_L], k)
    s[RHS_S].compute_at(s[OUT_L], k)
    s[LHS_L].compute_at(s[OUT_L], k)
    s[RHS_L].compute_at(s[OUT_L], k)    

    m, k = LHS_S.op.axis
    # t = s[LHS_S].fuse(m, k)
    # # t = 1024*1024
    # t, vi = s[LHS_S].split(t, factor=4)
    # # t = 1024*1024/4 = 1024*256, vi = 4
    # t, tx = s[LHS_S].split(t, factor=32)
    # # t = 1024*256/32 = 1024*8, tx = 32
    # _, ty = s[LHS_S].split(t, factor=32)
    # # _ = 1024*8/32 = 32*8,ty = 32
    ty, xi = s[LHS_S].split(m, nparts=scale)
    s[LHS_S].bind(ty, te.thread_axis("threadIdx.y"))
    # tx, xi = s[LHS_S].split(k, nparts=32)
    # s[LHS_S].bind(tx, te.thread_axis("threadIdx.x"))
    # s[LHS_S].vectorize(vi)


    k, n = RHS_S.op.axis
    # t = s[RHS_S].fuse(k, n)
    # # t = 1024*1024
    # t, vi = s[RHS_S].split(t, factor=4)
    # # t = 1024*1024/4 = 1024*256, vi = 4
    # t, tx = s[RHS_S].split(t, factor=32)
    # # t = 1024*256/32 = 1024*8, tx = 32
    # _, ty = s[RHS_S].split(t, factor=32)
    # _ = 1024*8/32 = 32*8,ty = 32
    # ty, xi = s[RHS_S].split(k, nparts=32)
    # s[RHS_S].bind(ty, te.thread_axis("threadIdx.y"))
    tx, xi = s[RHS_S].split(n, nparts=scale)
    s[RHS_S].bind(tx, te.thread_axis("threadIdx.x"))
    # s[RHS_S].vectorize(vi)

    mod = tvm.lower(s, [LHS, RHS, OUT], simple_mode=True, name="gemm")
    print(mod.astext(show_meta_data=False))

    # correctness
    def check_device(device):
        dev = tvm.device(device, 0)
        if not dev.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Device %s" % device)

        f = tvm.build(s, [LHS, RHS, OUT], target=device, name="gemm")
        # print("source code:\n", f.get_source())
        # launch the kernel.
        m, n, k = M, N, K
        lhs_np = np.random.uniform(size=(m, k)).astype(LHS.dtype)
        rhs_np = np.random.uniform(size=(k, n)).astype(RHS.dtype)
        lhs = tvm.nd.array(lhs_np, dev)
        rhs = tvm.nd.array(rhs_np, dev)
        out = tvm.nd.array(np.zeros((m, n), dtype=OUT.dtype), dev)
        f(lhs, rhs, out)
        tvm.testing.assert_allclose(out.numpy(), np.dot(lhs_np, rhs_np), rtol=1e-5)

        num_flops = M * N * K + (M * N * K - 1)
        num_runs = 10
        timer_f = f.time_evaluator(f.entry_name, dev, number=num_runs)
        t = timer_f(lhs, rhs, out).mean
        GFLOPS = num_flops / (t * 1e3) / 1e6
        print("average time cost of %d runs = %g ms, %g GFLOPS." % (num_runs, t * 1e3, GFLOPS))

    check_device("cuda")

if __name__ == "__main__":
    test_gemm()
