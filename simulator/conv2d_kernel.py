from numpy import broadcast
from util import *
from mem import *
import numpy as np

class Conv2dKernel():
  def __init__(self, DEBUG=0):
    self.DEBUG = DEBUG
    return

  def PRINT(self, string):
    if self.DEBUG==1: print(string)
    # print(string)
    return

  # =============================================================== #
  #                       algorithm. conv_loop
  # =============================================================== #
  def conv_loop(self, params, f_val, w_val, bias_val, res):
    # f[N,Hi,Wi,Ci]
    # w[R,S,Ci,Co]
    # o[N,Ho,Wo,Co]
    N,Hi,Wi,Ci,R,S,Co,Ho,Wo,strides,padding,dilation = params
    Sh = strides[0]
    Sw = strides[1]
    
    f_pad = np.pad(f_val, ((0, 0), (1,1), (1,1), (0, 0)), mode='constant')

    for n in range(N):
      for ho in range(Ho):
        for wo in range(Wo):
          for co in range(Co):
            res[n,ho,wo,co] = 0
            for r in range(R):
              for s in range(S):
                for ci in range(Ci):
                  res[n,ho,wo,co] += f_pad[n,(ho*Sh+r),(wo*Sw+s),ci] * w_val[r,s,ci,co]
            res[n,ho,wo,co] += bias_val[0,0,0,co]
    # import pdb;pdb.set_trace()
    res = np.maximum(np.zeros_like(res), res)
    
    return res
  # =============================================================== #
  #                       0. step0
  # =============================================================== #
  def step0(self, params, f_val, w_val, bias_val, res):
    util = Util()
    mem = MEM()
    # f[N,Hi,Wi,Ci]
    # w[R,S,Ci,Co]
    # o[N,Ho,Wo,Co]
    N,Hi,Wi,Ci,R,S,Co,Ho,Wo,strides,padding,dilation = params
    Sh = strides[0]
    Sw = strides[1]
    Dh = dilation[0]
    Dw = dilation[1]
    Pt = padding[0]
    Pd = padding[1]
    Pl = padding[2]
    Pr = padding[3]
    # R = R if Dh==1 else Dh*(R-1)+1
    # S = S if Dw==1 else Dw*(S-1)+1
    # Hpi = (Ho-1)*Sh+R-Pt-Pd
    # Wpi = (Wo-1)*Sw+S-Pl-Pr
    Hpi = Ho+Pt+Pd
    Wpi = Wo+Pl+Pr

    A = f_val.flatten()
    W = w_val.flatten()
    bias = bias_val.flatten()
    conv2d_nhwc_output = res.flatten()
    
    # compute PaddedInput
    f_pad = mem.new([N*Hpi*Wpi*Ci], "zero")
    
    for n in range(N):
      for hi in range(Hpi):
        for wi in range(Wpi):
          for ci in range(Ci):
            if (hi>=1 and hi<(Hpi-Pd)) and (wi>=1 and wi<(Wpi-Pr)): 
              f_pad[n*Hpi*Wpi*Ci + hi*Wpi*Ci + wi*Ci + ci] = A[n*Hi*Wi*Ci + hi*Wi*Ci - Pt*Wi*Ci + wi*Ci -Pl*Ci + ci]
            else:
              f_pad[n*Hpi*Wpi*Ci + hi*Wpi*Ci + wi*Ci + ci] = 0
    # compute W[h, w, i, o]
    compute = mem.new([R*S*Ci*Co], "zero")
    for r in range(R):
      for s in range(S):
        for ci in range(Ci):
          for co in range(Co):
            compute[r*S*Ci*Co + s*Ci*Co + ci*Co + co] = W[r*S*Ci*Co + s*Ci*Co + ci*Co + co]
    # compute conv2d_nhwc_output
    for n in range(N):
      for ho in range(Ho):
        for wo in range(Wo):
          for co in range(Co):
            conv2d_nhwc_output[n*Ho*Wo*Co + ho*Wo*Co + wo*Co + co] = 0
            for r in range(R):
              for s in range(S):
                for ci in range(Ci):
                  # res[n,ho,wo,co] += f_pad[n,(ho*Sh+r),(wo*Sw+s),ci] * w_val[r,s,ci,co]
                  o_index = n*Ho*Wo*Co + ho*Wo*Co + wo*Co + co
                  f_index = n*Hpi*Wpi*Ci + (ho*Sh+r)*Wpi*Ci + (wo*Sw+s)*Ci + ci
                  w_index = r*S*Ci*Co + s*Ci*Co + ci*Co + co
                  
                  conv2d_nhwc_output[o_index] += f_pad[f_index] * compute[w_index]
            conv2d_nhwc_output[n*Ho*Wo*Co + ho*Wo*Co + wo*Co + co] += bias[co]
    # compute max
    for n in range(N):
      for ho in range(Ho):
        for wo in range(Wo):
          for co in range(Co):
            conv2d_nhwc_output[n*Ho*Wo*Co + ho*Wo*Co + wo*Co + co] = max(conv2d_nhwc_output[n*Ho*Wo*Co + ho*Wo*Co + wo*Co + co], 0)
    # import pdb;pdb.set_trace()
    res = np.reshape(conv2d_nhwc_output, (N,Ho,Wo,Co))
    return res
