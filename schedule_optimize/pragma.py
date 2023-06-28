import tvm
from tvm import te
import pdb

def test_pragma_1():
  n = 1024
  m = 1024
  A = te.placeholder((n, m), name='A')
  k = te.reduce_axis((0, n), name='k')
  l = te.reduce_axis((0, m), name = 'l')

  B = te.compute((n,), lambda i: te.sum(A[i, l], axis=l), name='B')

  s = te.create_schedule(B.op)

  ko, ki = s[B].split(B.op.reduce_axis[0], factor=4)

  print(tvm.lower(s, [A, B], simple_mode=True))
  print("---------cutting line---------")

  s1 = s[B].pragma(ki, "unroll")
  # s1 = s[B].pragma(ko, "vectorize") # error
  pdb.set_trace()
  print(tvm.lower(s, [A, B], simple_mode=True))

def test_pragma_2():
  # m = te.var("m")
  # l = te.var("l")
  m = 16
  l = 16
  A = te.placeholder((m, l), name="A")
  B = te.compute((m, l), lambda i, j: A[i, j], name="B")

  s = te.create_schedule(B.op)
  xo, xi = s[B].split(B.op.axis[0], 8)

  print(tvm.lower(s, [B, A], simple_mode=True))
  print("---------cutting line---------")

  s[B].pragma(xo, "auto_unroll_max_step", 2)
  print(tvm.lower(s, [B, A], simple_mode=True))

  # mod = schedule_to_module(s, [A, A1])
  # assert isinstance(mod["main"], tvm.tir.PrimFunc)

def test_pragma_3():
  nn = 1024
  n = tvm.runtime.convert(nn)
  A = te.placeholder((n,), name="A")
  B = te.placeholder((n,), name="B")
  AA = te.compute((n,), lambda *i: A(*i), name="A")
  BB = te.compute((n,), lambda *i: B(*i), name="B")
  T = te.compute(A.shape, lambda *i: AA(*i) + BB(*i), name="T")
  C = te.compute(A.shape, lambda *i: T(*i), name="C")
  s = te.create_schedule(C.op)
  xo, xi = s[C].split(C.op.axis[0], factor=64)
  s[C].bind(xo, te.thread_axis("threadIdx.x"))
  xi1, xi2 = s[C].split(xi, factor=4)
  s[C].parallel(xi2)
  print(tvm.lower(s, [C, A, B], simple_mode=True))
  print("---------cutting line---------")

  import pdb;pdb.set_trace()
  # s[C].pragma(xi1, "rewrite_thread_axis", {"threadIdx.x": te.thread_axis("threadIdx.x"), "val": 64})
  s[C].pragma(xi1, "rewrite_thread_axis", "threadIdx.x->threadIdx.x, tval->1024, func->floordiv, fval->64")
  print(tvm.lower(s, [C, A, B], simple_mode=True))

  s[C].pragma(xi1, "parallel_launch_point")
  s[C].pragma(xi2, "parallel_stride_pattern")
  s[C].pragma(xi2, "parallel_barrier_when_finish")


if __name__ == "__main__":
  # test_pragma_1()
  # test_pragma_2()
  test_pragma_3()