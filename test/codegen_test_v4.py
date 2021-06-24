import tvm
import tvm.testing
from tvm import te
import numpy
import timeit
import pdb


###=======================================###
### tvm.build dot parallel 
###=======================================###
def te_dot_parallel(target, dtype, dev):
  M = 64
  N = 64
  K = 64
  k = te.reduce_axis((0,K),'k')
  A = te.placeholder((M,K),name='A')
  B = te.placeholder((K,N),name='B')
  C = te.compute((M,N),lambda m,n:te.sum(A[m,k]*B[k,n], axis=k),name='dot')
  D = te.compute((M,N),lambda m,n:te.max(C[m,n], 0), name="relu")

  bn = 32
  packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
  C = te.compute(
    (M, N),
    lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
    name="C",
  )

  s = te.create_schedule(C.op)
  CC = s.cache_write(C, "global")
  xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
  s[CC].compute_at(s[C], yo)
  xc, yc = s[CC].op.axis
  (k,) = s[CC].op.reduce_axis
  ko, ki = s[CC].split(k, factor=4)
  s[CC].reorder(ko, xc, ki, yc)
  s[CC].unroll(ki)
  s[CC].vectorize(yc)

  # parallel
  s[C].parallel(xo)

  x, y, z = s[packedB].op.axis
  s[packedB].vectorize(z)
  s[packedB].parallel(x)

  func = tvm.build(s, [A, B, C], target=target, name="mmult")
  assert func

  print("--------------------tir:\n")
  print(tvm.lower(s, [A, B, C], simple_mode=True))
  if target == "c":
    print("--------------------code:\n", func.get_source())
  return

###=======================================###
### tvm.build dot cache write
###=======================================###
def te_dot_cache_write(target, dtype, dev):
  M = 64
  N = 64
  K = 64
  k = te.reduce_axis((0,K),'k')
  A = te.placeholder((M,K),name='A')
  B = te.placeholder((K,N),name='B')
  C = te.compute((M,N),lambda m,n:te.sum(A[m,k]*B[k,n], axis=k),name='dot')
  D = te.compute((M,N),lambda m,n:te.max(C[m,n], 0), name="relu")

  bn = 32
  packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
  C = te.compute(
    (M, N),
    lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
    name="C",
  )

  s = te.create_schedule(C.op)

  # Allocate write cache
  CC = s.cache_write(C, "global")

  xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)

  # Write cache is computed at yo
  s[CC].compute_at(s[C], yo)

  # New inner axes
  xc, yc = s[CC].op.axis

  (k,) = s[CC].op.reduce_axis
  ko, ki = s[CC].split(k, factor=4)
  s[CC].reorder(ko, xc, ki, yc)
  s[CC].unroll(ki)
  s[CC].vectorize(yc)

  x, y, z = s[packedB].op.axis
  s[packedB].vectorize(z)
  s[packedB].parallel(x)

  func = tvm.build(s, [A, B, C], target=target, name="mmult")
  assert func

  print("--------------------tir:\n")
  print(tvm.lower(s, [A, B, C], simple_mode=True))
  if target == "c":
    print("--------------------code:\n", func.get_source())
  return

###=======================================###
### tvm.build dot array packing
###=======================================###
def te_dot_array_packing(target, dtype, dev):
  M = 64
  N = 64
  K = 64
  k = te.reduce_axis((0,K),'k')
  A = te.placeholder((M,K),name='A')
  B = te.placeholder((K,N),name='B')
  C = te.compute((M,N),lambda m,n:te.sum(A[m,k]*B[k,n], axis=k),name='dot')
  D = te.compute((M,N),lambda m,n:te.max(C[m,n], 0), name="relu")

  bn = 32
  packedB = te.compute((N / bn, K, bn), lambda x, y, z: B[y, x * bn + z], name="packedB")
  C = te.compute(
    (M, N),
    lambda x, y: te.sum(A[x, k] * packedB[y // bn, k, tvm.tir.indexmod(y, bn)], axis=k),
    name="C",
  )

  s = te.create_schedule(C.op)

  xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
  (k,) = s[C].op.reduce_axis
  ko, ki = s[C].split(k, factor=4)

  s[C].reorder(xo, yo, ko, xi, ki, yi)
  s[C].vectorize(yi)

  x, y, z = s[packedB].op.axis
  s[packedB].vectorize(z)
  s[packedB].parallel(x)

  func = tvm.build(s, [A, B, C], target=target, name="mmult")
  assert func

  print("--------------------tir:\n")
  print(tvm.lower(s, [A, B, C], simple_mode=True))
  if target == "c":
    print("--------------------code:\n", func.get_source())
  return

###=======================================###
### tvm.build dot permutation
###=======================================###
def te_dot_permutation(target, dtype, dev):
  M = 64
  N = 64
  K = 64
  k = te.reduce_axis((0,K),'k')
  A = te.placeholder((M,K),name='A')
  B = te.placeholder((K,N),name='B')
  C = te.compute((M,N),lambda m,n:te.sum(A[m,k]*B[k,n], axis=k),name='dot')
  D = te.compute((M,N),lambda m,n:te.max(C[m,n], 0), name="relu")

  s = te.create_schedule(C.op)
  bn = 32
  xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
  (k,) = s[C].op.reduce_axis
  ko, ki = s[C].split(k, factor=4)

  # re-ordering
  s[C].reorder(xo, yo, ko, xi, ki, yi)
  s[C].vectorize(yi)

  func = tvm.build(s, [A, B, C], target=target, name="mmult")
  assert func

  print("--------------------tir:\n")
  print(tvm.lower(s, [A, B, C], simple_mode=True))
  if target == "c":
    print("--------------------code:\n", func.get_source())
  return

###=======================================###
### tvm.build dot vectorize
###=======================================###
def te_dot_vec(target, dtype, dev):
  M = 64
  N = 64
  K = 64
  k = te.reduce_axis((0,K),'k')
  A = te.placeholder((M,K),name='A')
  B = te.placeholder((K,N),name='B')
  C = te.compute((M,N),lambda m,n:te.sum(A[m,k]*B[k,n], axis=k),name='dot')
  D = te.compute((M,N),lambda m,n:te.max(C[m,n], 0), name="relu")

  s = te.create_schedule(C.op)
  bn = 32
  xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
  (k,) = s[C].op.reduce_axis
  ko, ki = s[C].split(k, factor=4)

  s[C].reorder(xo, yo, ko, ki, xi, yi)

  # Vectorization
  s[C].vectorize(yi)

  func = tvm.build(s, [A, B, C], target=target, name="mmult")
  assert func

  print("--------------------tir:\n")
  print(tvm.lower(s, [A, B, C], simple_mode=True))
  if target == "c":
    print("--------------------code:\n", func.get_source())
  return


###=======================================###
### tvm.build dot tile
###=======================================###
def te_dot_tile(target, dtype, dev):
  # The size of the matrix
  # (M, K) x (K, N)
  # You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
  M = 64
  K = 64
  N = 64

  # The default tensor type in tvm
  dtype = "float32"

  # Algorithm
  k = te.reduce_axis((0, K), "k")
  A = te.placeholder((M, K), name="A")
  B = te.placeholder((K, N), name="B")
  C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

  bn = 32
  s = te.create_schedule(C.op)
  pdb.set_trace()
  # Blocking by loop tiling
  xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
  (k,) = s[C].op.reduce_axis
  ko, ki = s[C].split(k, factor=4)

  # Hoist reduction domain outside the blocking loop
  s[C].reorder(xo, yo, ko, ki, xi, yi)

  func = tvm.build(s, [A, B, C], target=target, name="mmult")
  assert func

  print("--------------------tir:\n")
  print(tvm.lower(s, [A, B, C], simple_mode=True))
  if target == "c":
    print("--------------------code:\n", func.get_source())
  return

###=======================================###
### tvm.build dot 
###=======================================###
def te_dot(target, dtype, dev):
  # The size of the matrix
  # (M, K) x (K, N)
  # You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
  M = 64
  K = 64
  N = 64

  # The default tensor type in tvm
  dtype = "float32"

  # Algorithm
  k = te.reduce_axis((0, K), "k")
  A = te.placeholder((M, K), name="A")
  B = te.placeholder((K, N), name="B")
  C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")

  # Default schedule
  s = te.create_schedule(C.op)
  func = tvm.build(s, [A, B, C], target=target, name="mmult")
  assert func

  if target == "llvm":
    # Random generated tensor for testing
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev)

    np_repeat = 100
    np_runing_time = timeit.timeit(
      setup="import numpy\n"
      "M = " + str(M) + "\n"
      "K = " + str(K) + "\n"
      "N = " + str(N) + "\n"
      'dtype = "float32"\n'
      "a = numpy.random.rand(M, K).astype(dtype)\n"
      "b = numpy.random.rand(K, N).astype(dtype)\n",
      stmt="answer = numpy.dot(a, b)",
      number=np_repeat,
    )
    print("Numpy running time: %f" % (np_runing_time / np_repeat))

    answer = numpy.dot(a.numpy(), b.numpy())


    c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev)
    func(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), answer, rtol=1e-5)

    evaluator = func.time_evaluator(func.entry_name, dev, number=1)
    print("Baseline: %f" % evaluator(a, b, c).mean)

  print("--------------------tir:\n")
  print(tvm.lower(s, [A, B, C], simple_mode=True))
  if target == "c":
    print("--------------------code:\n", func.get_source())




if __name__ == "__main__":
  dtype = "float32"
  # using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
  # To get the best performance, please change the following line
  # to llvm -mcpu=core-avx2, or specific type of CPU you use
  #target = "llvm"
  target = "c"
  dev = tvm.device(target, 0)
  #te_dot(target, dtype, dev)
  te_dot_tile(target, dtype, dev)
  #te_dot_vec(target, dtype, dev)
  #te_dot_permutation(target, dtype, dev)
  #te_dot_array_packing(target, dtype, dev)
  #te_dot_cache_write(target, dtype, dev)
  #te_dot_parallel(target, dtype, dev)
