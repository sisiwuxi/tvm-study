# namespace for tensor expression utility
import tvm
from tvm import te
from tvm.ir.module import IRModule
import numpy as np

# declare the computation using the expression API
A = te.placeholder((128, ), name="A")
B = te.placeholder((128, ), name="B")
C = te.compute((128,), lambda i: A[i] + B[i], name="C")

# create a function with the specified list of arguments. 
func = te.create_prim_func([A, B, C])
# mark that the function name is main
func = func.with_attr("global_symbol", "main")
ir_mod_from_te = IRModule({"main": func})
# print(ir_mod_from_te.script())

sch = tvm.tir.Schedule(ir_mod_from_te)
# print(type(sch))

rt_mod = tvm.build(sch.mod, target="llvm")  # The module for CPU backends.
# print(type(rt_mod))
# func = rt_mod["main"]
# print(func)

a = tvm.nd.array(np.arange(128, dtype="float32"))
b = tvm.nd.array(np.ones(128, dtype="float32"))
c = tvm.nd.empty((128,), dtype="float32")
# print(sch.mod.script())
rt_mod["main"](a, b, c)
# func(a, b, c)
print(a)
print(b)
print(c)