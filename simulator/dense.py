
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

    # # =====================  1. step0  ===================== #
    # # c_m, c_n = C.op.axis  # 128,1008
    # # c_bm, c_m = s[C].split(c_m, factor=block_m)  # 8,16
    # # c_bn, c_n = s[C].split(c_n, factor=block_n)  # 63,16
    # # s[C].reorder(c_bm, c_bn, c_m, c_n)  # 8,63,16,16
    # # s[C].bind(c_bm, te.thread_axis("blockIdx.y"))  # 8
    # # s[C].bind(c_bn, te.thread_axis("blockIdx.x"))  # 63
    # # calculate [128,16,2048] in 1 thread
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma
    # res = dense.step0(param)
    # util.check_result(res_std, res, " step0 ")

    # # =====================  1.0 step1_0_fuse_dense  ===================== #
    # # c_t = s[C].fuse(c_m, c_n)  # 16*16
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma
    # res = dense.step1_0_fuse_dense(param)
    # util.check_result(res_std, res, " step1_0_fuse_dense ")

    # # =====================  1.0 step1_0  ===================== #
    # # c_t, c_tv = s[C].split(c_t, factor=vec)  # 3584,2
    # # c_t, c_tx = s[C].split(c_t, factor=block_tx)  # 56,64
    # # c_t, c_ty = s[C].split(c_t, factor=block_ty)  # 14,4
    # # c_t, c_tz = s[C].split(c_t, factor=block_tz)  # 2,7
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec
    # res = dense.step1_0(param)
    # util.check_result(res_std, res, " step1_0 ")

    # # =====================  1 step1  ===================== #
    # # s[C].bind(c_tx, te.thread_axis("threadIdx.x"))  # 64
    # # s[C].bind(c_ty, te.thread_axis("threadIdx.y"))  # 4
    # # s[C].bind(c_tz, te.thread_axis("threadIdx.z"))  # 7
    # # s[C].vectorize(c_tv)
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec
    # res = dense.step1(param)
    # util.check_result(res_std, res, " step1 ")

    # # =====================  2_0 step2_0  ===================== #
    # # cs_m, cs_n = CS.op.axis  # 128/2,1008/9 = 64,112
    # # s[CS].storage_align(cs_m, CS_align - 1, CS_align)  # align 16
    # # cs_m, cs_mi = s[CS].split(cs_m, factor=wmma_m)  # 1,16
    # # cs_m, cs_ty = s[CS].split(cs_m, factor=block_ty)  # 1,1
    # # cs_n, cs_ni = s[CS].split(cs_n, factor=wmma_n)  # 1,16
    # # cs_n, cs_tx = s[CS].split(cs_n, factor=block_tz)  # 1,1
    # # s[CS].reorder(cs_m, cs_n, cs_ty, cs_tx, cs_mi, cs_ni)  # 1,1,1,1,16,16
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec
    # res = dense.step2_0(param)
    # util.check_result(res_std, res, " step2_0 ")

    # # =====================  2_1 step2_1  ===================== #
    # # cs_m, cs_n = CS.op.axis  # 128/2,1008/9 = 64,112
    # # s[CS].storage_align(cs_m, CS_align - 1, CS_align)  # align 16
    # # cs_m, cs_mi = s[CS].split(cs_m, factor=wmma_m)  # 1,16
    # # cs_m, cs_ty = s[CS].split(cs_m, factor=block_ty)  # 1,1
    # # cs_n, cs_ni = s[CS].split(cs_n, factor=wmma_n)  # 1,16
    # # cs_n, cs_tx = s[CS].split(cs_n, factor=block_tz)  # 1,1
    # # s[CS].reorder(cs_m, cs_n, cs_ty, cs_tx, cs_mi, cs_ni)  # 1,1,1,1,16,16
    # # s[CS].bind(cs_ty, te.thread_axis("threadIdx.y"))  # 16
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec
    # res = dense.step2_1(param)
    # util.check_result(res_std, res, " step2_1 ")

    # # =====================  2 step2  ===================== #
    # # T_local_shared[64,128] aligned & reuse A_shared[16,2048] ???
    # # s[CS].bind(cs_tx, te.thread_axis("threadIdx.z"))  # 16
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec
    # res = dense.step2(param)
    # util.check_result(res_std, res, " step2 ")

    # # =====================  3 step3  ===================== #
    # # T_local_shared[64,128] aligned & reuse A_shared[16,2048] ???
    # # cf_m, cf_n = CF.op.axis  # 16,16
    # # cf_m, cf_mi = s[CF].split(cf_m, factor=wmma_m)  # 1,16
    # # cf_n, cf_ni = s[CF].split(cf_n, factor=wmma_n)  # 1,16
    # # cf_k = CF.op.reduce_axis[0]  # 2048
    # # cf_k, cf_kii = s[CF].split(cf_k, factor=wmma_k)  # 128,16
    # # cf_ko, cf_ki = s[CF].split(cf_k, factor=k_factor)  # 16,8
    # # s[CF].reorder(cf_ko, cf_ki, cf_m, cf_n, cf_mi,
    # #               cf_ni, cf_kii)  # 16,8,1,1,16,16,16
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor
    # res = dense.step3(param)
    # util.check_result(res_std, res, " step3 ")


    # # =====================  4.0 step4_0  ===================== #
    # # s[CS].compute_at(s[C], c_bn)
    # # s[CF].compute_at(s[CS], cs_tx)  # m,n,ty,tx
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor
    # res = dense.step4_0(param)
    # util.check_result(res_std, res, " step4_0 ")
    # return

    # # =====================  4.1 step4_1  ===================== #
    # # s[AS].compute_at(s[CF], cf_ko)
    # # s[AF].compute_at(s[CF], cf_m)  # ko,ki,m,n
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor
    # res = dense.step4_1(param)
    # util.check_result(res_std, res, " step4_0 ")
    # return

    # # =====================  4 step4  ===================== #
    # # s[BS].compute_at(s[CF], cf_ko)
    # # s[BF].compute_at(s[CF], cf_m)
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor
    # res = dense.step4(param)
    # util.check_result(res_std, res, " step4 ")
    # return

    # # =====================  5 step5  ===================== #
    # # af_m, af_k = AF.op.axis  # 16,2048
    # # af_m, af_mi = s[AF].split(af_m, factor=wmma_m)  # 1,16
    # # af_k, af_ki = s[AF].split(af_k, factor=wmma_k)  # 128,16
    # # s[AF].reorder(af_m, af_k, af_mi, af_ki)  # 1,128,16,16
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor
    # res = dense.step5(param)
    # util.check_result(res_std, res, " step5 ")
    # return

    # # =====================  6 step6  ===================== #
    # # bf_n, bf_k = BF.op.axis  # 16,2048
    # # bf_n, bf_ni = s[BF].split(bf_n, factor=wmma_n)  # 1,16
    # # bf_k, bf_ki = s[BF].split(bf_k, factor=wmma_k)  # 128,16
    # # s[BF].reorder(bf_n, bf_k, bf_ni, bf_ki)  # 1,128,16,16
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor
    # res = dense.step6(param)
    # util.check_result(res_std, res, " step6 ")
    # return

    # # =====================  7 step7_0  ===================== #
    # # as_m, as_k = AS.op.axis  # 16,16
    # # s[AS].storage_align(as_m, AS_align - 1, AS_align)  # align 16
    # # as_t = s[AS].fuse(as_m, as_k)  # 16*16
    # # as_t, as_tv = s[AS].split(as_t, factor=vec)  # 256,1
    # # as_t, as_tx = s[AS].split(as_t, factor=block_tx)  # 4,64
    # # as_t, as_ty = s[AS].split(as_t, factor=block_ty)  # 4,1
    # # as_t, as_tz = s[AS].split(as_t, factor=block_tz)  # 4,1
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # alignment = [136,136,128]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor,alignment
    # res = dense.step7_0(param)
    # util.check_result(res_std, res, " step7_0 ")
    # return

    # # =====================  7 step7  ===================== #
    # # s[AS].bind(as_tx, te.thread_axis("threadIdx.x"))  # 64
    # # s[AS].bind(as_ty, te.thread_axis("threadIdx.y"))  # 1
    # # s[AS].bind(as_tz, te.thread_axis("threadIdx.z"))  # 1
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # alignment = [136,136,128]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor,alignment
    # res = dense.step7(param)
    # util.check_result(res_std, res, " step7 ")
    # return


    # # =====================  8.0 step8_0  ===================== #
    # # bs_n, bs_k = BS.op.axis  # 16,16
    # # s[BS].storage_align(bs_n, BS_align - 1, BS_align)  # align 16
    # # bs_t = s[BS].fuse(bs_n, bs_k)  # 16*16
    # # bs_t, bs_tv = s[BS].split(bs_t, factor=vec)  # 256,1
    # # bs_t, bs_tx = s[BS].split(bs_t, factor=block_tx)  # 4,64
    # # bs_t, bs_ty = s[BS].split(bs_t, factor=block_ty)  # 4,1
    # # bs_t, bs_tz = s[BS].split(bs_t, factor=block_tz)  # 4,1
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # alignment = [136,136,128]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor,alignment
    # res = dense.step8_0(param)
    # util.check_result(res_std, res, " step8_0 ")
    # return

    # # =====================  8 step8  ===================== #
    # # s[BS].bind(bs_tx, te.thread_axis("threadIdx.x"))  # 64
    # # s[BS].bind(bs_ty, te.thread_axis("threadIdx.y"))  # 1
    # # s[BS].bind(bs_tz, te.thread_axis("threadIdx.z"))  # 1
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # alignment = [136,136,128]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor,alignment
    # res = dense.step8(param)
    # util.check_result(res_std, res, " step8 ")
    # return


    # # =====================  9 step9  ===================== #
    # # s[AF].tensorize(af_mi, intrin_bi_wmma_load_matrix_gemm((wmma_m, wmma_n, wmma_k), "A", strides_src=AS_stride, strides_dst=AF_stride),)
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # alignment = [136,136,128]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor,alignment
    # res = dense.step9(param)
    # util.check_result(res_std, res, " step9 ")
    # return

    # # =====================  10 step10  ===================== #
    # # s[BF].tensorize(bf_ni, intrin_bi_wmma_load_matrix_gemm((wmma_m, wmma_n, wmma_k), "B", strides_src=BS_stride, strides_dst=BF_stride),)
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # alignment = [136,136,128]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor,alignment
    # res = dense.step10(param)
    # util.check_result(res_std, res, " step10 ")
    # return

    # # =====================  11 step11  ===================== #
    # # s[CF].tensorize(cf_mi, intrin_bi_wmma_gemm((wmma_m, wmma_n, wmma_k),bi_wmma_compute, input_scope="local", strides_A=AF_stride, strides_B=BF_stride, strides_C=CF_stride,),)
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # alignment = [136,136,128]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor,alignment
    # res = dense.step11(param)
    # util.check_result(res_std, res, " step11 ")
    # return

    # # =====================  12 step12  ===================== #
    # # s[CF].tensorize(cf_mi, intrin_bi_wmma_gemm((wmma_m, wmma_n, wmma_k),bi_wmma_compute, input_scope="local", strides_A=AF_stride, strides_B=BF_stride, strides_C=CF_stride,),)
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(9,2,1)
    # block = [64,4,7]
    # wmma = [16,16,16]
    # vec = 2
    # k_factor = 8
    # alignment = [136,136,128]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor,alignment
    # res = dense.step12(param)
    # util.check_result(res_std, res, " step12 ")
    # return

    # # =====================  13 step13  ===================== #
    # # s[AS].double_buffer()
    # # s[BS].double_buffer()
    # res = mem.new(res_shape, "zero")
    # block_loop = [1,1,1] # grid=(7,4,1)
    # block = [64,2,9]
    # wmma = [16,16,16]
    # vec = 1
    # k_factor = 2
    # wmma_m = wmma[0] # 16
    # wmma_n = wmma[1] # 16
    # wmma_k = wmma[2] # 16
    # offset = 0
    # offsetCS = 0
    # AS_align = k_factor*wmma_k + offset
    # BS_align = k_factor*wmma_k + offset
    # CS_align = block_loop[2]*block[2]*wmma_n + offsetCS
    # alignment = [AS_align,BS_align,CS_align]
    # param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor,alignment
    # res = dense.step13(param)
    # util.check_result(res_std, res, " step13 ")
    # return

    # =====================  14 step14  ===================== #
    # s[AS].double_buffer()
    # s[BS].double_buffer()
    res = mem.new(res_shape, "zero")
    block_loop = [1,1,1] # grid=(7,4,1)
    block = [64,2,9]
    wmma = [16,16,16]
    vec = 1
    k_factor = 2
    wmma_m = wmma[0] # 16
    wmma_n = wmma[1] # 16
    wmma_k = wmma[2] # 16
    offset = 0
    offsetCS = 0
    AS_align = k_factor*wmma_k + offset
    BS_align = k_factor*wmma_k + offset
    CS_align = block_loop[2]*block[2]*wmma_n + offsetCS
    alignment = [AS_align,BS_align,CS_align]
    param = tile_shape['o'], lhs, rhs, res, block_loop, block, wmma, vec, k_factor,alignment
    res = dense.step14(param)
    util.check_result(res_std, res, " step14 ")
    return

if __name__ == '__main__':
    test()