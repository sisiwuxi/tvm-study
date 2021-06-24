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

def test_assign_func(target):
  n = 10
  A = te.placeholder((n, n))
  B = te.compute((n, n), lambda i, j: A[i, j])
  s = te.create_schedule(B.op)
  f = tvm.build(s, [A, B], target)
  if target == "llvm":
    dev = tvm.cpu(0)
    a = tvm.nd.array(np.random.randint(0, 2, size=(n,n)).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.randint(0, 2, size=(n,n)).astype(A.dtype), dev)
    f(a,b)
    b_np = a 
    tvm.testing.assert_allclose(b.numpy(), b_np)
  return f

def test_multiple_func(target):
  nn = 1024
  n = tvm.runtime.convert(nn)
  A = te.placeholder((n,), name="A")
  B = te.placeholder((n,), name="B")
  C = te.placeholder((n,), name="C")
  D = te.compute(A.shape, lambda *i: A(*i) * B(*i), name="D")
  E = te.compute(D.shape, lambda *i: D(*i) + C(*i), name="E")
  s = te.create_schedule(E.op)
  f = tvm.build(s, [A, B, C], target)
  return f

def test_scale_func(target):
  n = 64
  A = te.placeholder((n,), name="A")
  scale = te.placeholder((), name="scale")
  k = te.reduce_axis((0, n), name="k")
  C = te.compute((), lambda: te.sum(A[k] * scale(), axis=k), name="C")
  D = te.compute((), lambda: C() + 1)
  s = te.create_schedule(D.op)
  # build and invoke the kernel.
  #pdb.set_trace()
  f = tvm.build(s, [A, scale, D], target)

  if target == "llvm":
    dev = tvm.cpu(0)
    # launch the kernel.
    a = tvm.nd.array(np.random.randint(0, 2, size=(n,)).astype(A.dtype), dev)
    sc = tvm.nd.array(np.random.randint(0, 2, size=()).astype(scale.dtype), dev)
    d = tvm.nd.empty((), D.dtype, dev)
    f(a, sc, d)
    d_np = np.sum(a.numpy()) * sc.numpy() + 1
    tvm.testing.assert_allclose(d.numpy(), d_np)

  return f

def relay_test_conv2d_fuse_before(target):
    """Test fusion case of conv2d"""
    def before(dshape):
        x = relay.var("x", shape=dshape)
        x = relay.add(x, relay.const(1, "float32"))
        y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=16)
        # this is the next dominator.
        y1 = relay.add(relay.const(1, "float32"), y)
        y = relay.add(y, y1)
        # second path
        z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
        z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(3, 3), padding=(1, 1), channels=16)
        # add can only be fused to z1
        z = relay.add(z2, z3)
        pdb.set_trace()
        return relay.Function(relay.analysis.free_vars(z), z)
    dshape = (1, 16, 64, 64)
    z = before(dshape)
    print("---------------------before---------------------")
    print("---------------------ir module:")
    print(z.astext())
    with relay.build_config(opt_level=0):
      f = relay.build(z, target)
    print("---------------------graph json:")
    print(f.graph_json)
    print("---------------------c code:")
    print(f.lib.get_source())
    return

def relay_test_conv2d_fuse_after(target):
    """Test fusion case of conv2d"""
    def after(dshape):
        x = relay.var("x", shape=dshape)
        x = relay.add(x, relay.const(1, "float32"))
        y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=16)
        # this is the next dominator.
        y1 = relay.add(relay.const(1, "float32"), y)
        y = relay.add(y, y1)
        # second path
        z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
        z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(3, 3), padding=(1, 1), channels=16)
        # add can only be fused to z1
        z = relay.add(z2, z3)
        z = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
        return relay.Function(relay.analysis.free_vars(z), z)
    dshape = (1, 16, 64, 64)
    z = after(dshape)
    print("---------------------after---------------------")
    print("---------------------ir module:")
    print(z.astext())
    with relay.build_config(opt_level=0):
      f = relay.build(z, target)
    print("---------------------graph json:")
    print(f.graph_json)
    print("---------------------c code:")
    print(f.lib.get_source())
    """
    zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=0))
    print("---------------------zz---------------------")
    ff = relay.build(zz, target)
    print(ff.graph_json)
    print(zz.astext())
    """
    return

def test_add_conv2d(target):
    """Test fusion case of conv2d"""
    def before(dshape):
        x = relay.var("x", shape=dshape)
        x = relay.add(x, relay.const(1, "float32"))
        y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=16)
        return relay.Function(relay.analysis.free_vars(y), y)
    dshape = (1, 16, 64, 64)
    y = before(dshape)
    print("---------------------z---------------------")
    pdb.set_trace()
    f = relay.build(y, target)
    print(f.graph_json)
    print(y.astext())
    return


def relay_test_add(target):
  x = relay.var('x', shape=(10, 10))
  f = relay.Function([x], x + x)
  pdb.set_trace()
  mod = tvm.IRModule({"main": f})
  # create a Relay VM.
  dev = tvm.cpu()
  executable = relay.vm.compile(mod, target)
  code, lib = executable.save()
  return 

def relay_test_add_conv(target):
  dshape = (1, 16, 64, 64)
  x = relay.var('x', shape=dshape)
  x = relay.add(x, relay.const(1, "float32"))
  y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=16)
  f = relay.Function(relay.analysis.free_vars(y), y)
  #pdb.set_trace()
  mod = tvm.IRModule({"main": f})
  # create a Relay VM.
  dev = tvm.cpu()
  executable = relay.vm.compile(mod, target)
  code, lib = executable.save()
  return

def relay_test_before_fuse(target):
  dshape = (1, 16, 64, 64)
  x = relay.var("x", shape=dshape)
  x = relay.add(x, relay.const(1, "float32"))
  y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=16)
  # this is the next dominator.
  y1 = relay.add(relay.const(1, "float32"), y)
  y = relay.add(y, y1)
  # second path
  z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
  z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(3, 3), padding=(1, 1), channels=16)
  # add can only be fused to z1
  z = relay.add(z2, z3)
  f = relay.Function(relay.analysis.free_vars(z), z)
  #pdb.set_trace()
  mod = tvm.IRModule({"main": f})
  # create a Relay VM.
  dev = tvm.cpu()
  executable = relay.vm.compile(mod, target)
  code, lib = executable.save()
  return

def relay_test_after_fuse(target):
  dshape = (1, 16, 64, 64)
  x = relay.var("x", shape=dshape)
  x = relay.add(x, relay.const(1, "float32"))
  y = relay.nn.conv2d(x, relay.var("w1"), kernel_size=(3, 3), padding=(1, 1), channels=16)
  # this is the next dominator.
  y1 = relay.add(relay.const(1, "float32"), y)
  y = relay.add(y, y1)
  # second path
  z2 = relay.nn.conv2d(y, relay.var("w2"), kernel_size=(1, 1), padding=(0, 0), channels=16)
  z3 = relay.nn.conv2d(y, relay.var("w3"), kernel_size=(3, 3), padding=(1, 1), channels=16)
  # add can only be fused to z1
  z = relay.add(z2, z3)
  zz = run_opt_pass(z, transform.FuseOps(fuse_opt_level=2))
  f = relay.Function(relay.analysis.free_vars(zz), zz)
  #pdb.set_trace()
  mod = tvm.IRModule({"main": f})
  # create a Relay VM.
  dev = tvm.cpu()
  executable = relay.vm.compile(mod, target)
  code, lib = executable.save()
  return


if __name__ == "__main__":
  target = "c"
  #target = "llvm"
  #f = test_assign_func(target)
  #f = test_multiple_func(target)
  #f = test_scale_func(target)
  #print(f.get_source())
  #relay_test_add_conv2d(target)
  #relay_test_conv2d_fuse(target)
  #relay_test_add(target)
  #relay_test_add_conv(target)
  #relay_test_before_fuse(target)
  #relay_test_after_fuse(target)
  relay_test_conv2d_fuse_before(target)
  #relay_test_conv2d_fuse_after(target)
