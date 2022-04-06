
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
 
    # # =====================  0. step0  ===================== #
    # res = mem.new(res_shape, "zero")
    # param = tile_shape['o'], lhs, rhs, res
    # res = dense.step0_dense(param)
    # util.check_result(res_std, res, " step0_dense ")

    # # =====================  1. step1_1  ===================== #
    # # c_bm,c_m = s[C].split(c_m,block_m)
    # # c_bn,c_n = s[C].split(c_n,block_n)
    # calculate [16,16,2048] in 1 block
    # res = mem.new(res_shape, "zero")
    # grid = [63,8,1] # N,M 
    # block = [64,1,1]
    # wmma = [16,16,16]
    # param = tile_shape['o'], lhs, rhs, res, grid, block, wmma
    # res = dense.step1_dense_1(param)
    # util.check_result(res_std, res, " step1_dense_1 ")

    # =====================  1. step1  ===================== #
    # c_bm,c_m = s[C].split(c_m,block_m)
    # c_bn,c_n = s[C].split(c_n,block_n)
    # calculate [64,16,2048] in 1 block
    res = mem.new(res_shape, "zero")
    grid = [1,1,1] # N,M [63,2,1]
    block = [64,4,1]
    wmma = [16,16,16]
    param = tile_shape['o'], lhs, rhs, res, grid, block, wmma
    res = dense.step1_dense(param)
    util.check_result(res_std, res, " step1_dense ")

    # # =====================  2. step2  ===================== #
    # # c_t = s[C].fuse(c_m,c_n)
    # res = mem.new(res_shape, "zero")
    # grid = [63,8,1] # N,M
    # block = [1,1,1]
    # param = tile_shape['o'], lhs, rhs, res, grid, block
    # res = dense.step2_dense(param)
    # util.check_result(res_std, res, " step2_dense ")

    # # =====================  3. step3  ===================== #
    # # c_t,c_tx = s[C].split(c_t,block_tx)
    # # c_t,c_ty = s[C].split(c_t,block_ty)
    # # c_t,c_tz = s[C].split(c_t,block_tz)
    # res = mem.new(res_shape, "zero")
    # grid = [63,8,1] # N,M
    # # block = [64,1,1]
    # block = [32,2,1]
    # param = tile_shape['o'], lhs, rhs, res, grid, block
    # res = dense.step3_dense(param)
    # util.check_result(res_std, res, " step3_dense ")

    # # =====================  4. step4  ===================== #
    # # s[C].bind(c_tx, te.thread_axis("threadIdx.x"))
    # res = mem.new(res_shape, "zero")
    # grid = [63,8,1] # N,M
    # block = [32,2,1]
    # param = tile_shape['o'], lhs, rhs, res, grid, block
    # res = dense.step4_dense(param)
    # util.check_result(res_std, res, " step4_dense ")

    # # =====================  5. step5  ===================== #
    # # cs_m, cs_mi = s[CS].split(cs_m, wmma_m)
    # # cs_m, cs_ty = s[CS].split(cs_m, wmma_m)
    # # cs_n, cs_ni = s[CS].split(cs_n, wmma_n)
    # # cs_n, cs_tx = s[CS].split(cs_n, wmma_n)
    # res = mem.new(res_shape, "zero")
    # grid = [63,8,1] # N,M
    # block = [32,2,1]
    # wmma = [16,16,16]
    # param = tile_shape['o'], lhs, rhs, res, grid, block, wmma
    # res = dense.step5_dense(param)
    # util.check_result(res_std, res, " step5_dense ")

    return

if __name__ == '__main__':
    test()