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
    M = 16
    N = 32
    K = 64
    m = te.var("n")
    m = tvm.runtime.convert(M)
    n = te.var("n")
    n = tvm.runtime.convert(N)
    k = te.var("n")
    k = tvm.runtime.convert(K)
    # Algorithm
    LHS = te.placeholder((m, k), name="LHS")
    RHS = te.placeholder((k, n), name="RHS")
    k = te.reduce_axis((0, k), name="k")
    OUT = te.compute((m, n), lambda i, j: te.sum(LHS[i, k] * RHS[k, j], axis=k), name="OUT")   
    s = te.create_schedule(OUT.op)

    num_thread = m*n
    # num_block = (num_thread + total_thread - 1)/total_thread
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")

    mo, no = OUT.op.axis
    # pdb.set_trace()
    # average time cost of 10 runs = 0.010648 ms, 6.15468 GFLOPS
    # s[OUT].bind(no, block_x)
    # s[OUT].bind(mo, thread_x)

    # average time cost of 10 runs = 0.0074786 ms, 8.763 GFLOPS
    s[OUT].bind(mo, block_x)
    s[OUT].bind(no, thread_x)

    # correctness
    def check_device(device):
        dev = tvm.device(device, 0)
        if not dev.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Device %s" % device)

        f = tvm.build(s, [LHS, RHS, OUT], device)
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
