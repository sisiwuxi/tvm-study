"""
#=======================================#
target: tvm -> factor
mode:   pull mode
#=======================================#

1
dtu codegen
 a
 dataflow = cmd_package
  dma: runtime, copy
  control loop
  datatype
  dtu.driver
  dtu.optimize
 b
 kernel = instrinsic
  extend convgen
  tensorize instrinsic

2
dtu runtime
 read_from_dtu_memory
 dtu_conv2d
 dtu_relu

3
test
 op
  dot_relu, conv_relu
 network
  resnet50 after relay optimize
"""

"""
#=======================================#
tutorial
#=======================================#
https://tvm.apache.org/docs/dev/relay_bring_your_own_codegen.html?highlight=codegen

dtu codegen class
VisitExpr_(const CallNode* call)
visitor functions
JIT
Register codegen
CSourceModule
runtime class
GetFunction
Run
Register a runtime creation API
SaveToBinary
LoadFromBinary
create
"""

"""
#=======================================#
# data type
#=======================================#
1
factor
 pointer, tf32

2
TVM
"""

"""
#=======================================#
# VisitExpr_() 
#  Node
#   TVM_REGISTER_NODE_TYPE + ATTR_FUNCTOR_DISPATCH
#=======================================#
1
fast
 Dtype, Int, String, Auto, void, NUll, struct, Vector, Num, Name, Str, Type, Attribute, Call, Include, Using, Assign, Decl, List, Subscript, Index, Op
 Add, Sub, Div, Lt, Compare, AugAssign, Return, FuncDef, Code, For, Lambda, Expr, NewLine
factor
 DType_, FloatType_, IntType_, Value_, MemType_, DRAMType_, SRAMType_, L1Type_, DMAType_, CDMAType_, SDMAType_, Alloc_, AllocMem_, AllocDMA_, OnEngine_, 
 Launch_, AsyncLoad_, AsyncStore_, WaitDMA_, CG_, WaitCG_, Call_, ProgDecl_, ProgDef_, For_, Main_, Func_, CFunc_, Dim_, 

2
TVM
   TupleNode, TupleGetItemNode, ConstantNode, LetNode
   ArrayNode
   tir:
    IntImmNode,FloatImmNode,StringImmNode,VarNode,SizeVarNode,
    IterVarNode,
    AddNode,SubNode,MulNode,DivNode,ModNode,FloorDivNode,FloorModNode,MinNode,MaxNode,GENode,GTNode,LENode,LTNode,EQNode,NENode,AndNode,OrNode,NotNode
    CastNode,CallNode,SelectNode

"""
