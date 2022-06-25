import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(A: T.Buffer[128, "float32"], 
             B: T.Buffer[128, "float32"], 
             C: T.Buffer[128, "float32"]):
        # extra annotations for the function
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in range(128):
            with T.block("C"):
                # declare a data parallel iterator on spatial domain
                vi = T.axis.spatial(128, i)
                C[vi] = A[vi] + B[vi]

sch = tvm.tir.Schedule(MyModule)
print(type(sch))

# 1. Let us first try to split the loops
# Get block by its name
block_c = sch.get_block("C")
# Get loops surronding the block
(i,) = sch.get_loops(block_c)
# Tile the loop nesting.
i_0, i_1, i_2 = sch.split(i, factors=[None, 4, 4])
print(sch.mod.script())

# 2. We can also reorder the loops. Now we move loop i_2 to outside of i_1.
sch.reorder(i_0, i_2, i_1)
print(sch.mod.script())

# 3. Finally, we can add hints to the program generator that we want to vectorize the inner most loop.
sch.parallel(i_0)
print(sch.mod.script())

# We can build and run the transformed program
# rt_mod = runtime module
rt_mod = tvm.build(sch.mod, target="llvm")  # The module for CPU backends.
print(type(rt_mod))
func = rt_mod["main"]
print(func)

a = tvm.nd.array(np.arange(128, dtype="float32"))
b = tvm.nd.array(np.ones(128, dtype="float32"))
c = tvm.nd.empty((128,), dtype="float32")
print(sch.mod.script())
# rt_mod["main"](a, b, c)
func(a, b, c)
print(a)
print(b)
print(c)


