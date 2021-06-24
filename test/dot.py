import tvm
import numpy
import timeit

#矩阵大小(M,K)x(K,N),可以自由尝试不同的形状，有时TVM优化优于MKL的numpy。
M = 1024
K = 1024
N = 1024
#TVM中默认张量类型
dtype = "float32"
#使用Intel AVX2（高级矢量扩展）ISA进行SIMD
#获得最佳新能要修改下一行为'llvm -mcpu=core-avx2'，或者指定你使用的其他CPU类型
#实测指定CPU以后，Opt6版本可以达到略高于MKL numpy的性能。
target = 'llvm'
ctx = tvm.context(target, 0)

# Random generated tensor for testing
a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), ctx)

#numpy测试Numpy running time: 0.004753ms
np_repeat = 100
np_runing_time = timeit.timeit(setup='import numpy\n'
                                     'M = ' + str(M) + '\n'
                                     'K = ' + str(K) + '\n'
                                     'N = ' + str(N) + '\n'
                                     'dtype = "float32"\n'
                                     'a = numpy.random.rand(M, K).astype(dtype)\n'
                                     'b = numpy.random.rand(K, N).astype(dtype)\n',
                               stmt='answer = numpy.dot(a, b)',
                               number=np_repeat)
print("Numpy running time: %f" % (np_runing_time / np_repeat))

#基准测试：Baseline: 2.174880ms
#算法
k = tvm.reduce_axis((0, K), 'k')
A = tvm.placeholder((M, K), name='A')
B = tvm.placeholder((K, N), name='B')
C = tvm.compute(
           (M, N),
           lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k),
           name='C')
#默认调度
c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), ctx)
func(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)

evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Baseline: %f' % evaluator(a, b, c).mean)
