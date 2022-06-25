import tvm
from tvm import te
from tvm.ir.module import IRModule
import numpy as np

M = 1024
K = 1024
N = 1024

# The default tensor type in tvm
dtype = "float32"

target = "llvm"
dev = tvm.device(target, 0)

# Algorithm
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

# Default schedule
func = te.create_prim_func([A, B, C])
func = func.with_attr("global_symbol", "main")
ir_module = IRModule({"main": func})
# print(ir_module.script())

a = tvm.nd.array(np.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), dev)
c = tvm.nd.array(np.zeros((M, N), dtype=dtype), dev)

# Baseline: evaluator = 2.161911 s
func = tvm.build(ir_module, target="llvm")  # The module for CPU backends.
func(a, b, c)
evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("Baseline: %f" % evaluator(a, b, c).mean)

# split
# block_size    16          32          64          128         256
# evaluator(s)  0.178348    0.131138    0.113738    0.104099    0.266442
sch = tvm.tir.Schedule(ir_module)
# print(type(sch))
print(sch.mod.script())
block_c = sch.get_block("C")
# Get loops surronding the block
(y, x, k) = sch.get_loops(block_c)
block_size = 128 # 16,32,64,128,256
yo, yi = sch.split(y, [None, block_size])
xo, xi = sch.split(x, [None, block_size])
sch.reorder(yo, xo, k, yi, xi)
# print(sch.mod.script())
func = tvm.build(sch.mod, target="llvm")  # The module for CPU backends.
c = tvm.nd.array(np.zeros((M, N), dtype=dtype), dev)
func(a, b, c)
evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("after transformation: %f" % evaluator(a, b, c).mean)
