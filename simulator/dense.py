
import numpy as np
from util import *
from mem import *
from dense_kernel import *
import math
import pdb

def test():
    util = Util()
    mem = MEM()
    # params
    # M, N, K    = [8, 63, 128]
    M, N, K    = [128, 1008, 2048]
    SM, SN, SK = [32,  32, 16]
    AM, AN, AK = [16,  16, 16]
    KM, KN, KK = [AM,  4,  16]
    CM, CN, CK = [AM,  8,  16]
    # tile
    tile_shape = {}
    tile_shape['o'] = M, N, K
    tile_shape['s'] = SM, SN, SK
    tile_shape['c'] = CM, CN, CK
    tile_shape['k'] = KM, KN, KK
    tile_shape['a'] = AM, AN, AK
    # define
    lhs_shape = [M, K]
    rhs_shape = [N, K]
    res_shape = [M, N]
    # init
    lhs = mem.new(lhs_shape, "rand")
    rhs = mem.new(rhs_shape, "rand")
    # lhs = mem.new(lhs_shape, "ones")
    # rhs = mem.new(rhs_shape, "ones")

    # golden
    res_std = mem.new(res_shape, "zero")
    res_std = lhs @ rhs.T
    # dense
    dense = DenseKernel()
    # # =====================  ser dense  ===================== #
    # res = mem.new(res_shape, "zero")
    # param = tile_shape['o'], lhs, rhs, res
    # res = dense.ser_dense(param)
    # util.check_result(res_std, res, " ser_dense ")
 
    # # =====================  ori  ===================== #
    # res = mem.new(res_shape, "zero")
    # param = tile_shape['o'], lhs, rhs, res
    # res = dense.ori_dense(param)
    # util.check_result(res_std, res, " ori_dense ")

    # # =====================  1. step0_1  ===================== #
    # # c_bm,c_m = s[C].split(c_m,block_m)
    # # c_bn,c_n = s[C].split(c_n,block_n)
    # calculate [16,16,2048] in 1 thread
    # res = mem.new(res_shape, "zero")
    # grid = [63,8,1] # N,M 
    # block = [64,1,1]
    # wmma = [16,16,16]
    # param = tile_shape['o'], lhs, rhs, res, grid, block, wmma
    # res = dense.step0_dense_1(param)
    # util.check_result(res_std, res, " step0_dense_1 ")

    # =====================  1. step0  ===================== #
    # c_m, c_n = C.op.axis  # 128,1008
    # c_bm, c_m = s[C].split(c_m, factor=block_m)  # 8,16
    # c_bn, c_n = s[C].split(c_n, factor=block_n)  # 63,16
    # s[C].reorder(c_bm, c_bn, c_m, c_n)  # 8,63,16,16
    # s[C].bind(c_bm, te.thread_axis("blockIdx.y"))  # 8
    # s[C].bind(c_bn, te.thread_axis("blockIdx.x"))  # 63
    # calculate [128,16,2048] in 1 thread
    res = mem.new(res_shape, "zero")
    block_loop = [1,1,1] # grid=(9,2,1)
    block = [64,4,7]
    wmma = [16,16,16]
    param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma
    res = dense.step0_dense(param)
    util.check_result(res_std, res, " step0_dense ")

    # # =====================  1.0 step1_0_fuse_dense  ===================== #
    # # c_t = s[C].fuse(c_m, c_n)  # 16*16
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma
    # res = dense.step1_0_fuse_dense(param)
    # util.check_result(res_std, res, " step1_0_fuse_dense ")

    # =====================  1.0 step1_0  ===================== #
    # c_t, c_tx = s[C].split(c_t, factor=block_tx)  # 4,64
    # c_t, c_ty = s[C].split(c_t, factor=block_ty)  # 4,1
    # c_t, c_tz = s[C].split(c_t, factor=block_tz)  # 4,1
    res = mem.new(res_shape, "zero")
    block_loop = [1,1,1] # grid=(9,2,1)
    block = [64,4,7]
    wmma = [16,16,16]
    vec = 2
    param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec
    res = dense.step1_0(param)
    util.check_result(res_std, res, " step1_0 ")

    # # =====================  1.1 step1_1  ===================== #
    # # s[C].bind(c_tx, te.thread_axis("threadIdx.x"))  # 64
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma
    # res = dense.step1_1(param)
    # util.check_result(res_std, res, " step1_1 ")

    # # =====================  1.2 step1_2  ===================== #
    # # s[C].bind(c_ty, te.thread_axis("threadIdx.y"))  # 1
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma
    # res = dense.step1_2(param)
    # util.check_result(res_std, res, " step1_2 ")

    # # =====================  1 step1  ===================== #
    # # s[C].bind(c_tz, te.thread_axis("threadIdx.z"))  # 4
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma
    # res = dense.step1(param)
    # util.check_result(res_std, res, " step1 ")

    # # =====================  5. step5  ===================== #
    # # cs_m, cs_mi = s[CS].split(cs_m, factor=wmma_m)
    # # cs_m, cs_ty = s[CS].split(cs_m, factor=block_ty)
    # # cs_n, cs_ni = s[CS].split(cs_n, factor=wmma_n)
    # # cs_n, cs_tx = s[CS].split(cs_n, factor=block_tz)
    # # s[CS].reorder(cs_m, cs_n, cs_ty, cs_tx, cs_mi, cs_ni)
    # # s[CS].bind(cs_ty, te.thread_axis("threadIdx.y"))
    # # s[CS].bind(cs_tx, te.thread_axis("threadIdx.z"))
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma
    # res = dense.step5_dense(param)
    # util.check_result(res_std, res, " step5_dense ")

    return

if __name__ == '__main__':
    test()