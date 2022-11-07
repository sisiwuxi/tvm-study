import torch
import torch.nn as nn
import numpy as np
import time
import ctypes

def run_time(func):
  def wrapper(*args, **kw):
    start = time.time()
    res = func(*args, **kw)
    end = time.time()
    print('{0} run time is {1} second'.format(func.__name__, (end - start)))
    return res
  return wrapper

@run_time
def conv_reference(F, W):
  # NHWC -> NCHW
  F_torch = torch.tensor(F).permute(0,3,1,2)
  
  # KKIO -> OIKK
  W_torch = torch.tensor(W).permute(3,2,0,1)
  
  # run convolution
  out = nn.functional.conv2d(F_torch, W_torch)
  
  # NCHW -> NHWC
  return out.permute(0,2,3,1).contiguous().numpy()

@run_time
def conv_naive(F, W):
  N,Hi,Wi,Ci = F.shape
  Kh,Kw,Ci,Co = W.shape
  out = np.zeros((N,Hi-Kh+1,Wi-Kw+1,Co))
  for n in range(N):
    for ci in range(Ci):
      for co in range(Co):
        for h in range(Hi-Kh+1):
          for w in range(Wi-Kw+1):
            for kh in range(Kh):
              for kw in range(Kw):
                out[n,h,w,co] += F[n,h+kh,w+kw,ci] * W[kh,kw,ci,co]
  return out

@run_time
def conv_matrix_mult_1x1(F, W):
  N,Hi,Wi,Ci = F.shape
  Kh,Kw,Ci,Co = W.shape
  out = np.zeros((N,Hi-Kh+1,Wi-Kw+1,Co))

  # out = F @ W[0,0]
  # print(W[0,0].shape)

  # equivalent
  import pdb;pdb.set_trace()
  FR = F.reshape(-1,Ci)
  out2 = (FR @ W[0,0])
  out = out2.reshape(F.shape[0], F.shape[1], F.shape[2], W.shape[3])
  print(FR.shape, W[0,0].shape, out2.shape, out.shape)
  return out

@run_time
def conv_matrix_mult(F, W):
  N,Hi,Wi,Ci = F.shape
  Kh,Kw,Ci,Co = W.shape
  out = np.zeros((N,Hi-Kh+1,Wi-Kw+1,Co))

  for kh in range(Kh):
    for kw in range(Kw):
      out += F[:,kh:kh+Hi-Kh+1,kw:kw+Wi-Kw+1,:] @ W[kh,kw]
  return out

@run_time
def conv_im2col(F, W):
  N,Hi,Wi,Ci = F.shape
  Kh,Kw,Ci,Co = W.shape
  Ns, Hs, Ws, Cs = F.strides
  out = np.zeros((N,Hi-Kh+1,Wi-Kw+1,Co))

  inner_dim = Kh*Kw*Ci
  A = np.lib.stride_tricks.as_strided(F, shape = (N, Hi-Kh+1, Wi-Kw+1, Kh, Kw, Ci),
                                      strides = (Ns, Hs, Ws, Hs, Ws, Cs)).reshape(-1,inner_dim)
  out = A @ W.reshape(-1, Co)
  return out.reshape(N,Hi-Kh+1, Wi-Kw+1,Co)

def test_conv2d():
  F = np.random.randn(10,32,32,8)
  W = np.random.randn(3,3,8,16)
  out = conv_reference(F,W)
  print(out.shape)

  out_naive = conv_naive(F,W)
  print(np.linalg.norm(out - out_naive))

  out_mm = conv_matrix_mult(F,W)
  print(np.linalg.norm(out - out_mm))

  out_im2col = conv_im2col(F,W)
  print(np.linalg.norm(out - out_im2col))
  return

def test_conv2dot():
  # F = np.random.randn(10,32,32,8)
  # W = np.random.randn(1,1,8,16)

  F = np.random.randn(1,56,56,64)
  W = np.random.randn(1,1,64,64)

  out = conv_reference(F,W)
  print(out.shape)

  out2 = conv_matrix_mult_1x1(F,W)
  print(np.linalg.norm(out - out2))

  # out_mm = conv_matrix_mult(F,W)
  # print(np.linalg.norm(out - out_mm))
  return

def test_conv2d_strides():
  n = 6
  A = np.arange(n**2, dtype=np.float32).reshape(n,n)
  # print(A, A.shape)
  # print(np.frombuffer(ctypes.string_at(A.ctypes.data, A.nbytes), dtype=A.dtype, count=A.size))
  # print(A.strides)

  # B = np.lib.stride_tricks.as_strided(A, shape=(3,3,2,2), strides=np.array((12,2,6,1))*4)
  # print(B, B.shape)
  # print(np.frombuffer(ctypes.string_at(B.ctypes.data, size=B.nbytes), B.dtype, B.size))
  # print(B.strides)

  # C = np.ascontiguousarray(B)
  # print(C, C.shape)
  # print(np.frombuffer(ctypes.string_at(C.ctypes.data, size=C.nbytes), C.dtype, C.size))
  # print(C.strides)

  B = np.lib.stride_tricks.as_strided(A, shape=(4,4,3,3), strides=4*(np.array((6,1,6,1))))
  print(B)
  # print(np.frombuffer(ctypes.string_at(B.ctypes.data, size=A.nbytes), B.dtype, A.size))
  # print(B.strides)

  C = B.reshape(16,9)
  print(C)
  print(C.strides)

  W = np.arange(9, dtype=np.float32).reshape(3,3)
  print(W)

  out = (C @ W.reshape(9)).reshape(4,4)
  print(out)
  return out

if __name__ == "__main__":
  test_conv2d()
  # test_conv2dot()
  # test_conv2d_strides()