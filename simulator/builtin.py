
import numpy as np
import pdb

class BuildIn():
  def __init__(self, DEBUG=0):
    self.DEBUG = DEBUG
    return

  def PRINT(self, string):
    if self.DEBUG==1: print(string)
    # print(string)
    return
  
  def buildin_lane_id(self):
    return 64
  
  def builtin_matrix_mad_f32x4_f32x4(self, MMA, MMB, MMC):
    A = MMA.reshape(16,16)
    B = MMB.reshape(16,16)
    C = MMC.reshape(16,16)
    C = A@B + C
    MMC = C.reshape(64,4)
    return MMC
  
  def syncthreads(self):
    return

  def ld_st():
    thread_extent = 64
    for tx in range(64):
      print("\ntx = %d\n"%tx)
      col_offset = tx % 16
      row_offset = tx // 16 
      for blk_id in range(0,4):
        row = blk_id*4 + row_offset
        print("input[%02d,%02d]=%03d -> out[%02d,%02d]"%(row,col_offset,row*16+col_offset, 0,blk_id))
    return