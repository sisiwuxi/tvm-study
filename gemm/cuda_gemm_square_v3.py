# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
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
    # graph
    # nn = 2048
    nn = 8
    n = te.var("n")
    n = tvm.runtime.convert(nn)
    m, l = n, n
    # Algorithm
    A = te.placeholder((l, n), name="A")
    B = te.placeholder((l, m), name="B")
    k = te.reduce_axis((0, l), name="k")
    C = te.compute((m, n), lambda ii, jj: te.sum(A[k, jj] * B[k, ii], axis=k), name="C")

    s = te.create_schedule(C.op)

    scale = 8
    num_thread = 8
    block_factor = scale * num_thread
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")

    yo, xo = C.op.axis
    s[C].bind(xo, block_x)
    s[C].bind(yo, thread_x)


    # correctness
    def check_device(device):
        dev = tvm.device(device, 0)
        if not dev.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Device %s" % device)
        # Generate CUDA Kernel
        # Finally we use TVM to generate and compile the CUDA kernel, and evaluate the
        # latency of gemm.
        f = tvm.build(s, [A, B, C], device)
        # launch the kernel.
        n, m, l = nn, nn, nn
        a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
        b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
        a = tvm.nd.array(a_np, dev)
        b = tvm.nd.array(b_np, dev)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), dev)
        f(a, b, c)
        tvm.testing.assert_allclose(c.numpy(), np.dot(b_np.T, a_np), rtol=1e-5)

        num_flops = 2 * nn * nn * nn
        num_runs = 10
        timer_f = f.time_evaluator(f.entry_name, dev, number=num_runs)
        t = timer_f(a, b, c).mean
        GFLOPS = num_flops / (t * 1e3) / 1e6
        print("average time cost of %d runs = %g ms, %g GFLOPS." % (num_runs, t * 1e3, GFLOPS))

    check_device("cuda")

if __name__ == "__main__":
    test_gemm()
