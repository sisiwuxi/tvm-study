from __future__ import absolute_import, print_function
from operator import floordiv


import tvm
import tvm.testing
from tvm import te
import numpy as np

import sys
sys.path.append('../..')
# from common import *

def reduction_cpu_1(A, B):
    M, K = A.shape
    for m in range(M):
        B[m] = 0
        for k in range(K):
            B[m] = B[m] + A[m][k]
    return B


def reduction_cpu_2(A, B):
    M, K = A.shape
    for io in range(M//32):
        for ii in range(32):
            m = io*32 + ii
            if m < M:
                B[m] = 0
            if m < M:
                for ko in range(K//16):
                    for ki in range(16):
                        k = ko*16 + ki
                        if k < K:
                            B[m] = B[m] + A[m][k]
    return B

def reduction_cpu_3(A, B):
    M, K = A.shape
    SK = 16
    B_rf = np.zeros((SK*M), np.int8)
    for ki in range(SK):
        for m in range(M):
            B_rf[ki*M + m] = 0
            for ko in range(K//SK):
                k = ko*SK + ki
                if k < K:
                    B_rf[ki*M + m]  = B_rf[ki*M + m] + A[m][k]
    for m in range(K):
        B[m] = 0
        for ki in range(SK):
            B[m] = B[m] + B_rf[ki*K + m]
    return B

# gpu
def reduction_cpu_4(A, B):
    M, K = A.shape
    SK = 16
    for m in range(M):
        B_rf_local = np.zeros((1), np.int64)
        red_buf0_shared = np.zeros((SK), np.int64)
        for ki in range(SK):
            B_rf_local[0] = 0
            for ko in range(floordiv(K+SK-1, SK)):        
                k = ko*SK + ki
                if k < K:
                    B_rf_local[0] = B_rf_local[0] + A[m][k]
            red_buf0_shared[ki] = B_rf_local[0]
        # import pdb;pdb.set_trace()
        # SK = 16
        for ki in range(SK):
            if ki < 8:
                w_8_0 = red_buf0_shared[ki] + red_buf0_shared[ki + 8]
                red_buf0_shared[ki] = w_8_0
        for ki in range(SK):
            if ki < 4:
                w_4_0 = red_buf0_shared[ki] + red_buf0_shared[ki + 4]
                red_buf0_shared[ki] = w_4_0
        for ki in range(SK):
            if ki < 2:
                w_2_0 = red_buf0_shared[ki] + red_buf0_shared[ki + 2]
                red_buf0_shared[ki] = w_2_0
        for ki in range(SK):
            if ki < 1:
                w_1_0 = red_buf0_shared[ki] + red_buf0_shared[ki + 1]
                red_buf0_shared[ki] = w_1_0
        for ki in range(SK):
            if ki == 0:
                B[m] = red_buf0_shared[0]
    return B

# cuda shfl do not need shared memory
def reduction_cpu_5(A, B):
    import math
    M, K = A.shape
    SK = 128
    reduceIter = int(math.log2(SK))
    print("reduceIter=",reduceIter)
    B_rf_local = np.zeros((1), np.int64)
    red_buf0_shared = np.zeros((SK), np.int64)    
    for m in range(M):
        for ki in range(SK):
            B_rf_local[0] = 0
            for ko in range(floordiv(K+SK-1, SK)):        
                k = ko*SK + ki
                if k < K:
                    B_rf_local[0] = B_rf_local[0] + A[m][k]
            red_buf0_shared[ki] = B_rf_local[0]
        # import pdb;pdb.set_trace()
        for r in range(1, 1+reduceIter, 1):
            middle = int(SK//math.pow(2,r))
            for ki in range(SK):
                if ki < middle:
                    w_0 = red_buf0_shared[ki] + red_buf0_shared[ki+middle]
                    red_buf0_shared[ki] = w_0
        for ki in range(SK):
            if ki == 0:
                B[m] = red_buf0_shared[0]
    return B

def reduction_cpu_6(A, B):
    import math
    M, K = A.shape
    SK = 64
    reduceIter = int(math.log2(SK))
    print("reduceIter=",reduceIter)
    B_rf_local = np.zeros((1), np.int64)
    red_buf0_shared = np.zeros((SK), np.int64)
    for m in range(M):
        for ki in range(SK):
            B_rf_local[0] = 0
            for ko in range(floordiv(K+SK-1, SK)):        
                k = ko*SK + ki
                if k < K:
                    B_rf_local[0] = B_rf_local[0] + A[m][k]
            red_buf0_shared[ki] = B_rf_local[0]
        # SK == parallel, can not change loop nest
        for ki in range(SK):
            if ki < (SK//2):
                for r in range(1, 1+reduceIter, 1):
                    middle = int(SK//math.pow(2,r))
                    w_0 = red_buf0_shared[ki] + red_buf0_shared[ki+middle]
                    red_buf0_shared[ki] = w_0
        for ki in range(SK):
            if ki == 0:
                B[m] = red_buf0_shared[0]
    return B

def reduction_cpu():
    M = 128
    K = 128
    A = np.random.randint(1, 10, (M, K), np.int8)
    B = np.zeros((M), np.int64)

    # B = reduction_cpu_1(A, B)

    # B = reduction_cpu_2(A, B)

    # B = reduction_cpu_3(A, B)

    # B = reduction_cpu_4(A, B)

    # B = reduction_cpu_5(A, B)

    B = reduction_cpu_6(A, B) # err


    np_golden = np.sum(A, axis=1)
    tvm.testing.assert_allclose(B, np_golden, rtol=1e-4)
    print("pass")
    return

USE_TX = 0

def check_device(device):
    m = te.var("m")
    k = te.var("k")
    A = te.placeholder((m, k), name="A")
    k = te.reduce_axis((0, k), "k")
    B = te.compute((m,), lambda m: te.sum(A[m, k], axis=k), name="B")

    s = te.create_schedule(B.op)
    # print(tvm.lower(s, [A, B], simple_mode=True))

    # # ---------- split k ---------- #
    # ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
    # xo, xi = s[B].split(B.op.axis[0], factor=32)
    # print(tvm.lower(s, [A, B], simple_mode=True))

    # # ---------- bind ---------- #
    # s[B].bind(xo, te.thread_axis("blockIdx.x"))
    # s[B].bind(xi, te.thread_axis("threadIdx.x"))
    # print(tvm.lower(s, [A, B], simple_mode=True))

    # ---------- rfactor ---------- #
    s = te.create_schedule(B.op)
    ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
    BF = s.rfactor(B, ki)
    # print(tvm.lower(s, [A, B], simple_mode=True))
    # print(s[B].op.body)

    xo, xi = s[B].split(s[B].op.axis[0], factor=32)
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    if USE_TX == 1:
        s[B].bind(xi, te.thread_axis("threadIdx.y"))
        tx = te.thread_axis("threadIdx.x")
        s[B].bind(s[B].op.reduce_axis[0], tx)
        s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
        s[B].set_store_predicate(tx.var.equal(0))
    else:
        s[B].bind(xi, te.thread_axis("threadIdx.x"))
        ty = te.thread_axis("threadIdx.y")
        s[B].bind(s[B].op.reduce_axis[0], ty)
        s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
        s[B].set_store_predicate(ty.var.equal(0))

    f = tvm.build(s, [A, B], device)
    print(tvm.lower(s, [A, B], simple_mode=True))
    """
    @main = primfn(A_1: handle, B_1: handle) -> ()
        attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
        buffers = {B: Buffer(B_2: Pointer(float32), float32, [m: int32], [stride: int32], type="auto"),
                    A: Buffer(A_2: Pointer(float32), float32, [m, k: int32], [stride_1: int32, stride_2: int32], type="auto")}
        buffer_map = {A_1: A, B_1: B} {
        attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = floordiv((m + 31), 32);
        allocate(B.rf: Pointer(local float32), float32, [1]), storage_scope = local;
        allocate(reduce_temp0: Pointer(local float32), float32, [1]), storage_scope = local;
        attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
        attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 16 {
            B.rf[0] = 0f32
            for (k.outer: int32, 0, floordiv((k + 15), 16)) {
            if @tir.likely((((blockIdx.x*32) + threadIdx.x) < m), dtype=bool) {
                if @tir.likely((((k.outer*16) + threadIdx.y) < k), dtype=bool) {
                B.rf[0] = ((float32*)B.rf[0] + (float32*)A_2[((((blockIdx.x*32) + threadIdx.x)*stride_1) + (((k.outer*16) + threadIdx.y)*stride_2))])
                }
            }
            }
            attr [meta[tir.CommReducer][0]] "reduce_scope" = @tir.reinterpret(0u64, dtype=handle);
            @tir.tvm_thread_allreduce(1u32, (float32*)B.rf[0], True, reduce_temp0, threadIdx.y, dtype=handle)
            if (threadIdx.y == 0) {
            B_2[(((blockIdx.x*32) + threadIdx.x)*stride)] = (float32*)reduce_temp0[0]
            }
        }
    }
    """

    M, K = 64, 128
    # dev = tvm.cuda(0)
    dev = tvm.device(device, 0)
    a = tvm.nd.array(np.random.uniform(size=(M, K)).astype(A.dtype), dev)
    b = tvm.nd.array(np.zeros(M, dtype=B.dtype), dev)
    f(a, b)
    num_runs = 10
    evaluator = f.time_evaluator(f.entry_name, dev, number=num_runs)
    t = evaluator(a, b).mean
    print("average time cost of %d iterations = %g ms.\n" %(num_runs, t * 1e3))
    tvm.testing.assert_allclose(b.numpy(), np.sum(a.numpy(), axis=1), rtol=1e-4)
    # save_source_code(f, t, os.getcwd())
    return


def reduction_cuda():
    m = te.var("m")
    k = te.var("k")
    A = te.placeholder((m, k), name="A")
    k = te.reduce_axis((0, k), "k")
    B = te.compute((m,), lambda m: te.sum(A[m, k], axis=k), name="B")

    s = te.create_schedule(B.op)
    # print(tvm.lower(s, [A, B], simple_mode=True))

    # ---------- rfactor ---------- #
    s = te.create_schedule(B.op)
    ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
    BF = s.rfactor(B, ki)
    # print(tvm.lower(s, [A, B], simple_mode=True))
    # print(s[B].op.body)

    xo, xi = s[B].split(s[B].op.axis[0], factor=32)
    s[B].bind(xo, te.thread_axis("blockIdx.x"))
    s[B].bind(xi, te.thread_axis("threadIdx.y"))
    tx = te.thread_axis("threadIdx.x")
    s[B].bind(s[B].op.reduce_axis[0], tx)
    s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
    s[B].set_store_predicate(tx.var.equal(0))
    fcuda = tvm.build(s, [A, B], "cuda")
    print(fcuda.imported_modules[0].get_source())

    fcuda = tvm.build(s, [A, B], device)
    print(tvm.lower(s, [A, B], simple_mode=True))

    M, K = 64, 128
    dev = tvm.cuda(0)
    a = tvm.nd.array(np.random.uniform(size=(nn, nn)).astype(A.dtype), dev)
    b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), dev)
    fcuda(a, b)
    tvm.testing.assert_allclose(b.numpy(), np.sum(a.numpy(), axis=1), rtol=1e-4)
    return


def reduction():
    for device, ctx in tvm.testing.enabled_targets():
        print(device, ctx)
        check_device(device)
    return


if __name__ == '__main__':
    reduction_cpu()
    # reduction()
    # reduction_cuda()