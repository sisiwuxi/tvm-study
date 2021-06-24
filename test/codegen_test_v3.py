import collections
import ctypes
import json
import sys

import tvm
import tvm.testing
from tvm import te
from tvm import topi
from tvm.contrib import utils
import numpy as np
import ctypes
import math
import re
import pytest
import pdb
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass

from tvm import relay, runtime
from tvm.relay import testing
import os
from os import path as osp
import sys
import numpy
import timeit

###=======================================###
### tvm.build dot vectorize
###=======================================###
def te_test_dot_vec(target):
  M = 64
  N = 64
  K = 64
  k = te.reduce_axis((0,K),'k')
  A = te.placeholder((M,K),name='A')
  B = te.placeholder((K,N),name='B')
  C = te.compute((M,N),lambda m,n:te.sum(A[m,k]*B[k,n], axis=k),name='dot')
  D = te.compute((M,N),lambda m,n:te.max(C[m,n], 0), name="relu")

  s = te.create_schedule(D.op)
  # tile 32x32
  bn = 32
  xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
  k, = s[C].op.reduce_axis
  ko, ki = s[C].split(k, factor=4)
  s[C].reorder(xo, yo, ko, ki, xi, yi)
  # vectorize
  s[C].vectorize(yi)

  tir_m = tvm.lower(s, [A,B,C,D], simple_mode=True, name='dot')
  #rt_m = tvm.build(tir_m, [A,B,C,D], target=target, name='dot')
  #assert rt_m

  print("--------------------tir:\n", tir_m.astext(show_meta_data=True))
  #print("--------------------code:\n", rt_m.get_source())
  return

###=======================================###
### tvm.build dot tile
###=======================================###
def te_test_dot_tile(target):
  M = 64
  N = 64
  K = 64
  k = te.reduce_axis((0,K),'k')
  A = te.placeholder((M,K),name='A')
  B = te.placeholder((K,N),name='B')
  C = te.compute((M,N),lambda m,n:te.sum(A[m,k]*B[k,n], axis=k),name='dot')
  D = te.compute((M,N),lambda m,n:te.max(C[m,n], 0), name="relu")

  s = te.create_schedule(D.op)
  # tile 32x32
  bn = 32
  xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
  k, = s[C].op.reduce_axis
  ko, ki = s[C].split(k, factor=4)
  s[C].reorder(xo, yo, ko, ki, xi, yi)

  tir_m = tvm.lower(s, [A,B,C,D], simple_mode=True, name='dot')
  rt_m = tvm.build(tir_m, [A,B,C,D], target=target, name='dot')
  assert rt_m

  print("--------------------tir:\n", tir_m.astext(show_meta_data=True))
  print("--------------------code:\n", rt_m.get_source())
  return

###=======================================###
### tvm.build dot 
###=======================================###
def te_test_dot(target):
  M = 64
  N = 64
  K = 64
  k = te.reduce_axis((0,K),'k')
  A = te.placeholder((M,K),name='A')
  B = te.placeholder((K,N),name='B')
  C = te.compute((M,N),lambda m,n:te.sum(A[m,k]*B[k,n], axis=k),name='dot')
  D = te.compute((M,N),lambda m,n:te.max(C[m,n], 0), name="relu")

  s = te.create_schedule(D.op)
  tir_m = tvm.lower(s, [A,B,C,D], simple_mode=True, name='dot')
  rt_m = tvm.build(tir_m, [A,B,C,D], target=target, name='dot')
  assert rt_m

  #print("tir:\n", tir_m.astext(show_meta_data=False))
  print("--------------------tir:\n", tir_m.astext(show_meta_data=True))
  print("--------------------code:\n", rt_m.get_source())
  return

###=======================================###
### relay test resnet
###=======================================###
def relay_test_resnet(target):
  dshape = (1, 3, 224, 224)
  resnet, params = relay.testing.resnet.get_workload(layers=18, batch_size=dshape[0], image_shape=dshape[1:])

  with relay.build_config(opt_level=2):
    graph,lib,params = relay.build_module.build(resnet, "c", params=params) 
  #print(resnet.astext(show_mata_data=False))
  print(resnet.astext())
  #print(resnet.get_source())  
  build_dir = osp.abspath("./")
  lib.save(osp.join(build_dir, "model.c"))
  return

###=======================================###
### relay test sisi
###=======================================###
def relay_test_sisi(target):
  dshape = (1, 28, 28)
  net, params = relay.testing.sisi.get_workload(batch_size=dshape[0], dtype="float32")

  with relay.build_config(opt_level=3):
  #with relay.build_config(opt_level=2):
  #with relay.build_config(opt_level=1):
  #with relay.build_config(opt_level=0):
    graph,lib,params = relay.build_module.build(net, "c", params=params) 
  print(net.astext())
  build_dir = osp.abspath("./")
  lib.save(osp.join(build_dir, "sisi.c"))
  return

###=======================================###
### relay.build dot + relu
###=======================================###
def relay_dot(target):
    M = 64
    N = 64
    K = 64
    dtype = "float32"
    A = relay.var("x", relay.TensorType((1,M,K), dtype))
    B = relay.var("y", relay.TensorType((1,K,N), dtype))
    C = relay.nn.batch_matmul(A,B)
    D = relay.nn.relu(C)
    z = relay.Function(relay.analysis.free_vars(D), D)
    print("---------------------relay_dot---------------------")
    print("---------------------ir module:")
    print(z.astext())
    pdb.set_trace()
    with relay.build_config(opt_level=0):
      f = relay.build(z, target)
    print("---------------------graph json:")
    print(f.graph_json)
    print("---------------------c code:")
    print(f.lib.get_source())
    return

###=======================================###
### relay.build dot fuse relu
###=======================================###
def relay_dot_fuse(target):
    M = 64
    N = 64
    K = 64
    dtype = "float32"
    A = relay.var("x", relay.TensorType((1,M,K), dtype))
    B = relay.var("y", relay.TensorType((1,K,N), dtype))
    C = relay.nn.batch_matmul(A,B)
    D = relay.nn.relu(C)
    z = relay.Function(relay.analysis.free_vars(D), D)
    z = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
    print("---------------------relay_dot_fuse---------------------")
    print("---------------------ir module:")
    print(z.astext())
    with relay.build_config(opt_level=0):
      f = relay.build(z, target)
    print("---------------------graph json:")
    print(f.graph_json)
    print("---------------------c code:")
    print(f.lib.get_source())
    return

if __name__ == "__main__":
  target = "c"
  #target = "llvm"
  #te_test_dot(target)
  #te_test_dot_tile(target)
  te_test_dot_vec(target)
  #relay_test_resnet(target)
  #relay_test_sisi(target)
  #relay_dot(target)
  #relay_dot_fuse(target)
