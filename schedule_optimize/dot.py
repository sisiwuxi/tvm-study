import tvm
from tvm import te
import pdb

M = 1024
N = 1024
K = 1024
k = te.reduce_axis((0,K),'k')
LHS = te.placeholder((M,K), name='LHS')
RHS = te.placeholder((K,N), name='RHS')
OUT = te.compute(
    (M,N),
    lambda x,y: te.sum(LHS[x,k] * RHS[k,y], axis=k),
    name = 'OUT'
)
s = te.create_schedule(OUT.op)
pdb.set_trace()
# ir module
# ir_m = tvm.ir.function.PrimFunc
# primfn(LHS_1: handle, RHS_1: handle, OUT_1: handle) -> ()
ir_m = tvm.lower(s,[LHS,RHS,OUT], simple_mode=True, name='mmult')
# tvm.ir.module.IRModule
# rt_m = Module(c, 2d533d8)
rt_m = tvm.build(ir_m,[LHS,RHS,OUT],target='c',name='mmult')

# print
print("tir:\n", ir_m.astext(show_meta_data=False))
print("source code:\n", rt_m.get_source())