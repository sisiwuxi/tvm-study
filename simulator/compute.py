import numpy as np
from builtin import *
from util import *
from mem import *

# Data-Level Parallelism in Vector, SIMD, and GPU Architectures

def BatchedMatrixMulKernelBlockingTC(params):
  bi = BuildIn()
  util = Util()
  mem = MEM()

  A,B,C,M,N,K = params
  BLOCK_SIZE_M = 256
  BLOCK_SIZE_N = 256
  BLOCK_SIZE_K = 32
  WARP_SIZE_M = 32
  WARP_SIZE_N = 64
  TC_SHARP = 16
  warpSize = 64 # double 32 at least for parallel? the number of thread
  LANEID = bi.buildin_lane_id()
  # Wv4f32arpCnt = 32
  blockIdx_x = 0
  blockIdx_y = 0
  blockIdx_z = 0
  threadIdx_x = 32
  threadIdx_y = 0
  threadIdx_z = 0

  BA = A.flatten()
  BB = B.flatten()
  BC = C.flatten()

  warpCnt = (BLOCK_SIZE_M // WARP_SIZE_M) * (BLOCK_SIZE_N // WARP_SIZE_N) # 256//32 * 256//64 = 8 * 4 = 32
  warpMCount = BLOCK_SIZE_M // 4 # 64
  warpKCount = BLOCK_SIZE_K // 4 # 8
  warpNCount = BLOCK_SIZE_N // WARP_SIZE_N # 4
  
  SM = mem.new((BLOCK_SIZE_K//TC_SHARP, warpMCount, warpSize), "zero") # [32//16, 64, 64] = [2, 64, 64]
  SN = mem.new((BLOCK_SIZE_N//TC_SHARP, warpKCount, warpSize), "zero") # [256//16, 8, 64] = [16, 8, 64]

  MMA = mem.new((warpSize,4), "zero")
  MMB = mem.new((warpSize,4), "zero")
  # 2*4 means 1 warp need 8 MMC as output; 4 means 4 float32 per thread in 1 warp which own 64 threads
  MMC = mem.new((WARP_SIZE_M//TC_SHARP, WARP_SIZE_N//TC_SHARP, warpSize, 4), "zero") # [32//16, 64//16, 64, 4] = [2, 4,64, 4]

  # warpId = threadIdx.x
  for warpId in range(warpCnt): # #warp = #warpM*#warpN = 8*4 = 32
    WCMI = warpId // warpNCount
    WCNI = warpId % warpNCount
    # print(warpId,":", WCMI, WCNI)
    for laneId in range(LANEID): # 64

      for k in range((K-1)//BLOCK_SIZE_K + 1): # 1

        blockLoadACnt = BLOCK_SIZE_M*BLOCK_SIZE_K // warpSize # 256*32//64 = 128
        warpLoadACnt = blockLoadACnt // warpCnt # 128//32 =  4
        LA = mem.new((warpLoadACnt), "zero") # 4
        ds_a_row_off = BLOCK_SIZE_M*blockIdx_y + laneId//TC_SHARP
        ds_a_col_off = k*BLOCK_SIZE_K + laneId&15
        # print(warpId, laneId,":[",end=" ")
        for i in range(warpLoadACnt): # 4
          WAMI = (warpId*warpLoadACnt + i)%warpMCount # 64
          WAKI = (warpId*warpLoadACnt + i)//warpMCount # 64
          row = WAMI*4 + ds_a_row_off
          colomn = WAKI*16 + ds_a_col_off
          # print(row*K + colomn, end=",")
          LA[i] = 0
          if (row < M and colomn < K):
            LA[i] = BA[row*K + colomn]
        # print("]")
      
        blockLoadBCnt = BLOCK_SIZE_K*BLOCK_SIZE_N // warpSize # 32*256//64 = 128
        warpLoadBCnt = blockLoadBCnt // warpCnt # 128//32 = 4
        LB = mem.new((warpLoadBCnt), "zero") # 4
        ds_b_row_off = k*BLOCK_SIZE_K + laneId//TC_SHARP
        ds_b_col_off = BLOCK_SIZE_N*blockIdx_x + laneId&15
        # print(warpId, laneId,":[",end=" ")
        for i in range(warpLoadBCnt): # 4
          WBKI = (warpId*warpLoadBCnt + i)%warpKCount # 64
          WBNI = (warpId*warpLoadBCnt + i)//warpKCount # 64
          row = WBKI*4 + ds_b_row_off
          colomn = WBNI*16 + ds_b_col_off
          # print(row*N + colomn, end=",")
          LB[i] = 0
          if (row < K and colomn < N):
            LB[i] = BB[row*N + colomn]
        # print("]")

        # print(warpId, laneId,":[",end=" ")
        for i in range(warpLoadACnt): # 4
          WAMI = (warpId * warpLoadACnt + i)%warpMCount
          WAKI = (warpId * warpLoadACnt + i)//warpMCount
          # print(WAKI,WAMI,laneId, end=", ")
          SM[WAKI][WAMI][laneId] = LA[i]
        # print("]")

        # print(warpId, laneId,":[",end=" ")
        for i in range(warpLoadBCnt): # 4
          WBKI = (warpId * warpLoadBCnt + i)%warpKCount
          WBNI = (warpId * warpLoadBCnt + i)//warpKCount
          # print(WBNI,WBKI,laneId, end=", ")
          SN[WBNI][WBKI][laneId] = LB[i]
        # print("]")
        # bi.syncthreads()

  # bi.syncthreads()
  # warpId = threadIdx.x
  for warpId in range(warpCnt): # #warp = #warpM*#warpN = 8*4 = 32
    WCMI = warpId // warpNCount
    WCNI = warpId % warpNCount
    for mi in range(WARP_SIZE_M // TC_SHARP): # 32//16 = 2
      for ni in range(WARP_SIZE_N // TC_SHARP): # 64//16 = 4
        for ki in range(BLOCK_SIZE_K//TC_SHARP): # 32//16 = 2
          for laneId in range(LANEID): # 64
            # print(warpId, laneId)
            # SM[WAKI][WAMI][laneId] = SM[2, 64, 64] 
            wami = (WCMI * (WARP_SIZE_M // TC_SHARP) + mi)*4
            MMA[laneId][0] = SM[ki][wami + 0][laneId]
            MMA[laneId][1] = SM[ki][wami + 1][laneId]
            MMA[laneId][2] = SM[ki][wami + 2][laneId]
            MMA[laneId][3] = SM[ki][wami + 3][laneId]
            # SN[WBNI][WBKI][laneId] = SN[16, 8, 64]
            wbni = WCNI * (WARP_SIZE_N // TC_SHARP) + ni
            MMB[laneId][0] = SN[wbni][ki*4 + 0][laneId]
            MMB[laneId][1] = SN[wbni][ki*4 + 1][laneId]
            MMB[laneId][2] = SN[wbni][ki*4 + 2][laneId]
            MMB[laneId][3] = SN[wbni][ki*4 + 3][laneId]
            # print("MMA[",ki,wami,laneId,"] MMB[",wbni,ki*4,laneId,"] ")
        # print()
        MMC[mi][ni][:][:] = bi.builtin_matrix_mad_f32x4_f32x4(MMA, MMB, MMC[mi][ni][:][:])
    bi.syncthreads()

  # warpId = threadIdx.x
  for warpId in range(warpCnt): # #warp = #warpM*#warpN = 8*4 = 32
    WCMI = warpId // warpNCount
    WCNI = warpId % warpNCount
    for laneId in range(LANEID): # 64
      # print(warpId, laneId)
      col_base = BLOCK_SIZE_N * blockIdx_x + WARP_SIZE_N * WCNI + (laneId&15)
      row_base = BLOCK_SIZE_M * blockIdx_y + WARP_SIZE_M * WCMI + laneId//TC_SHARP
      for mi in range(0, WARP_SIZE_M, TC_SHARP):
        for ni in range(0, WARP_SIZE_N, TC_SHARP):
          for m_idx in range(4):
            col = col_base + ni
            row = row_base + mi + m_idx*4
            if (col < N and row < M):
              BC[row*N + col] = MMC[mi//TC_SHARP][ni//TC_SHARP][laneId][m_idx]
  import pdb;pdb.set_trace()
  C = BC.reshape(M,N)
  return C

def test():
  util = Util()
  mem = MEM()
  M,N,K = 256,256,32
  # define
  lhs_shape = [M, K]
  rhs_shape = [K, N]
  res_shape = [M, N]
  A = mem.new(lhs_shape, "rand")
  B = mem.new(rhs_shape, "rand")
  C = mem.new(res_shape, "zero")
  params = A,B,C,M,N,K
  BatchedMatrixMulKernelBlockingTC(params)
  return

if __name__ == '__main__':
  test()