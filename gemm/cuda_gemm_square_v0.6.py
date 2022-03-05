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
"""Example code to do square matrix multiplication."""
import tvm
from tvm import te
import os
from tvm.contrib import nvcc
from tvm.contrib import spirv
import numpy as np
import tvm.testing
import pdb
from tvm.contrib import tedd

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
    LHS_L = s.cache_read(LHS_S, "local", [OUT])
    RHS_S = s.cache_read(RHS, "shared", [OUT])
    RHS_L = s.cache_read(RHS_S, "local", [OUT])
    OUT_L = s.cache_write(OUT, "local")

    # grid[4,4,1] thread[32,32,1]
    scale = 32
    num_thread = 4
    vthread = 2
    block_factor = scale*num_thread

    m, n = OUT.op.axis

    # bm=8, tm=128
    bm, tm = s[OUT].split(m, factor=block_factor)
    # bn=8, tm=32, mi=4
    tm, mi = s[OUT].split(tm, factor=num_thread)
    # bn=8, tm=32, vm=2, mi=2
    vm, mi = s[OUT].split(mi, factor=vthread)
    # bn=8, tn=128
    bn, tn = s[OUT].split(n, factor=block_factor)
    # bn=8, tn=32, ni=4
    tn, ni = s[OUT].split(tn, factor=num_thread)
    # bn=8, tn=32, vn=2, ni=2
    vn, ni = s[OUT].split(ni, factor=vthread)

    s[OUT].bind(bm, te.thread_axis("blockIdx.x"))
    s[OUT].bind(bn, te.thread_axis("blockIdx.y"))
    s[OUT].bind(tm, te.thread_axis("threadIdx.x"))
    s[OUT].bind(tn, te.thread_axis("threadIdx.y"))
    s[OUT].bind(vm, te.thread_axis("vthread"))
    s[OUT].bind(vm, te.thread_axis("vthread"))
    # average time cost of 10 runs = 562.47 ms, 3.81795 GFLOPS.
    s[OUT].reorder(bm, bn, tm, tn, mi, ni)
    s[OUT_L].compute_at(s[OUT], tn)

    # average time cost of 10 runs = 562.515 ms, 3.81765 GFLOPS.
    # s[OUT].reorder(bm, bn, tn, tm, mi, ni)
    # s[OUT_L].compute_at(s[OUT], tm)
    # average time cost of 10 runs = 513.807 ms, 4.17956 GFLOPS.
    # s[OUT_L].compute_at(s[OUT], mi)
    # average time cost of 10 runs = 471.304 ms, 4.55648 GFLOPS.
    # s[OUT_L].compute_at(s[OUT], ni)

    m, n = OUT_L.op.axis
    tk, ki = s[OUT_L].split(k, factor=scale)
    bk, ki = s[OUT_L].split(ki, factor=1)
    # s[OUT_L].reorder(tk, bk, ki, m, n)
    s[LHS_S].compute_at(s[OUT_L], tk)
    s[RHS_S].compute_at(s[OUT_L], tk)    
    s[LHS_L].compute_at(s[OUT_L], bk)
    s[RHS_L].compute_at(s[OUT_L], bk)
    
    m, k = LHS_S.op.axis
    tm, mi = s[LHS_S].split(m, nparts=scale)
    tk, ki = s[LHS_S].split(k, nparts=scale)
    s[LHS_S].bind(tm, te.thread_axis("threadIdx.y"))
    s[LHS_S].bind(tk, te.thread_axis("threadIdx.x"))

    k, n = RHS_S.op.axis
    tk, ki = s[RHS_S].split(k, nparts=scale)
    tn, ni = s[RHS_S].split(n, nparts=scale)
    s[RHS_S].bind(tk, te.thread_axis("threadIdx.y"))
    s[RHS_S].bind(tn, te.thread_axis("threadIdx.x"))

    # pdb.set_trace()
    # print(tvm.lower(s, [LHS, RHS, OUT], simple_mode=True))
    mod = tvm.lower(s, [LHS, RHS, OUT], simple_mode=True, name="gemm")
    print(mod.astext(show_meta_data=False))

    # Tensor Expression Debug Display (TEDD)
    tedd.viz_dataflow_graph(s, dot_file_path="/tmp/dfg.dot")
    tedd.viz_schedule_tree(s, dot_file_path="/tmp/scheduletree.dot")
    s = s.normalize()
    tedd.viz_schedule_tree(s, dot_file_path="/tmp/scheduletree2.dot")
    tedd.viz_itervar_relationship_graph(s, dot_file_path="/tmp/itervar.dot")

    # correctness
    def check_device(device):
        dev = tvm.device(device, 0)
        if not dev.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Device %s" % device)

        f = tvm.build(s, [LHS, RHS, OUT], target=device, name="gemm")
        # print("source code:\n", f.get_source())
        dev_module = f.imported_modules[0]
        print("-----GPU code-----")
        print(dev_module.get_source())

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
