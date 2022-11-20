from __future__ import print_function
import sys
sys.path.append("./python")
import needle as ndl
from util import *
"""
While needle is designed as a minimalist framework, it contains a comprehensive the bells and whistles 
of standard deep learning frameworks.

Read and think about the relation between Tensor, array_api and underlying NDArray.
Think about how gradient are implemented.
"""

def create_a_needle_tensor():
  """
  creates a new Tensor y by adding a constant scalar to x
  """
  x = ndl.Tensor([1, 2, 3], dtype="float32")
  print("y from add_scalar")
  y = ndl.add_scalar(x, 1)
  print("y = ", y)
  print(y.numpy())
  print("y_1 from +")
  y_1 = x + 1
  print("y_1 = ", y_1)
  print(y_1.numpy())
  return

def computational_graph():
  """
  When running array computations, needle not only executes the arithmetic operations, 
  but also creates a computational graph along the way.
  y = exp(v1)*(exp(v1)+1)
  """
  v1 = ndl.Tensor([0], dtype="float32")
  v2 = ndl.exp(v1)
  v3 = v2 + 1
  v4 = v2 * v3

  print("v4.op =", v4.op)
  print("v4.inputs =", v4.inputs)
  print(v4.inputs[0] is v2 and v4.inputs[1] is v3)
  print("v4.inputs[0].inputs[0].op =", v4.inputs[0].inputs[0].op)
  print("v4.cached_data =", v4.cached_data)
  print("v3.op =", v3.op)
  print("v3.op.__dict__ =", v3.op.__dict__)
  print("v3.inputs =", v3.inputs)
  # v3's op class also contains a field that stores the scalar constant
  # class AddScalar(TensorOp):
  #   def __init__(self, scalar):
  #       self.scalar = scalar
  print("v3.op.__dict__ =", v3.op.__dict__)

  print_node(v4, "v4")
  print_node(v3, "v3")
  print_node(v2, "v2")
  print_node(v1, "v1")
  return

def executing_computation():
  # Now let us take a deeper look at what happens when we run an array operation. 
  # Specifically, what happens when we run ndl.exp
  x1 = ndl.Tensor([3], dtype="float32")
  x2 = ndl.Tensor([4], dtype="float32")
  # the calling path
  #   autograd.TensorOp.__call__ calls into
  #   autograd.Tensor.make_from_op calls into
  #   autograd.Tensor.realize_cached_data calls into
  #   x3.op.compute = ops.EWiseMul.compute
  # key points
  #   constructs the computational graph node
  #   The actual computation won't happen until realize_cached_data is called
  x3 = x1 * x2
  print("x3 = ", x3)
  # The following code can find the location of the compute implementation
  print("x3.op.compute.__code__ = ", x3.op.compute.__code__)
  # Here array_api simply points to numpy: import numpy as array_api
  # In the later lectures, we will replace array_api with our own implementation of NDArray
  # Each of the Value node stores a cached_data field that corresponds to the computed data. 
  # Because we are using numpy as our array api, the cached_data is a numpy.ndarray.
  print("type(x3.cached_data) = ",type(x3.cached_data))
  print("x3.cached_data = ",x3.cached_data)
  return

def lazy_evaluation():
  # We also support a lazy evaluation mode. In this case, we do not compute cached_data right away. 
  # But cached_data will be computed once we need the actual result (when we call x3.data, x3.numpy() or othercases).
  ndl.autograd.LAZY_MODE = True
  x1 = ndl.Tensor([3], dtype="float32")
  x2 = ndl.Tensor([4], dtype="float32")
  x3 = x1 * x2
  # We can see that x3's cached_data field is not yet readily available. But as soon as we call x3.numpy(), 
  # a call to x3.realize_cached_data() will get triggered to compute the actual value.
  print(x3.cached_data is None)
  print("x3.numpy() = ", x3.numpy())
  print(x3.cached_data)
  # By default we use eager evaluation mode that always directly realizes the computation, 
  # but lazy evaluation can also be helpful for some advanced optimizations later.
  return

def reverse_mode_ad():
  """
  Now we are ready to talk about reverse mode AD. As a recap from last lecture, we will need to traverse 
  the computational graph in reverse topological order, and construct the new adjoint nodes(Tensors).
  forward
  \|/
      v1     1
      |      |
  exp v2 ->+ v3
        \   /
          x
         v4
          |
          y
  v1 = 0
  v2 = exp(0) = 1
  v3 = v2+1 = 2
  v4 = v2xv3 = 1x2 = 2
  
  backward
  /|\
      v'1
      x|
      v'2 <- v'2->3
      +|        |id
      v'2->4    v'3
        x\    x/
          v'4
           |id
        out_grad
  v'4 = 1
  v'2->4 = v'4*v3 = 1*2 = 2    
  v'3 = v'4*v2 = 1*1 = 1
  v'2->3 = v'3 = 1
  v'2 = v'2->3 + v'2->4 = 1+2 = 3
  v'1 = v'2 * exp(v1) = 3*exp(0) = 3*1 = 3
  """
  # You will need to complete the implementations in autograd.py to enable the backward function. 
  # That computes the gradient and store them in the grad field of each input Tensor.
  v1 = ndl.Tensor([0], dtype="float32")
  # Exp
  v2 = ndl.exp(v1)
  # AddScalar
  v3 = v2 + 1
  # EWiseMul
  v4 = v2 * v3
  # Each op have a gradient function, that defines how to propagate adjoint back into its inputs(as partial adjoints). 
  # We can look up the gradient implementation of v4.op as follows(impl of gradient for multiplication)
  print(v4.op.gradient.__code__)
  print("v4.op = ", v4.op)
  # The gradient function defines a single step to propagate the output adjoint to partial adjoints of its inputs.
  out_grad = ndl.Tensor([1], dtype="float32")
  v4_grad = v4.op.gradient(out_grad, v4)
  print("v4_grad.inputs = ", out_grad, " -> gradient = ", v4_grad)
  # v4_grad[0].inputs = out_grad * rhs = out_grad * v3 = (needle.Tensor([1.]), needle.Tensor([2.]))
  v2_4_grad = v4_grad[0]
  print_node(v2_4_grad, "v2_4_grad")
  print("v2_4_grad.inputs = ", v2_4_grad.inputs, " -> gradient = ", v2_4_grad)
  # v4_grad[1].inputs = out_grad * lhs = out_grad * v2 = (needle.Tensor([1.]), needle.Tensor([1.]))
  v3_grad = v4_grad[1]
  print_node(v2_4_grad, "v3_grad")
  print("v3_grad.inputs = ", v3_grad.inputs, " -> gradient = ", v3_grad)
  v2_3_grad = v3.op.gradient(v3_grad, v3)
  print_node(v2_3_grad, "v2_3_grad")
  print("v2_3_grad.inputs = ", v2_3_grad.inputs, " -> gradient = ", v2_3_grad)
  v2_grad = v2_3_grad + v2_4_grad
  print_node(v2_grad, "v2_grad")
  print("v2_grad.inputs = ", v2_grad.inputs, " -> gradient = ", v2_grad)
  v1_grad = v2.op.gradient(v2_grad, v2)
  print_node(v1_grad, "v1_grad")
  print("v1_grad.inputs = ", v1_grad.inputs, " -> gradient = ", v1_grad)

  # v4_grad_value_only = v4.op.gradient_as_tuple(out_grad, v4)
  return


def sum_loss():
  """
  long chain computation graph
  0->1->2->...->99->100
     |  |  ...  |
     1  1  ...  1
  """   
  x = ndl.Tensor([1], dtype="float32")
  sum_loss = ndl.Tensor([0], dtype="float32")
  for i in range(100):
    sum_loss += x*x
  print_node(sum_loss, "sum_loss")
  print(sum_loss)
  print(sum_loss.inputs[0].inputs[0])
  # and so on, we have to carrying the entire chain of computations across different iterations
  # how to cost as little memory as possible ?

  # ===== detach ===== #
  # stop_gradient, no autograd
  # detach the computational graph
  # useful when you know that you no longer need to perform automatic differentiations
  # or you only need a value
  sum_loss_detach = ndl.Tensor([0], dtype="float32")
  for i in range(100):
    sum_loss_detach = (sum_loss_detach + x*x).detach()
  print_node(sum_loss_detach, "sum_loss")
  print(sum_loss_detach)
  print(sum_loss_detach.inputs) # none

  return

def test_power_scalar():
  # PowerScalar: Op raise a tensor to an (integer) power
  print("=========PowerScalar=======")
  # compute
  v1 = ndl.Tensor([3, 2], dtype="float32")
  v2 = ndl.power_scalar(v1, 2)
  print(v2)
  # gradient
  out_grad = ndl.Tensor([1, 1], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ", v2_grad.inputs, " -> gradient = ", v2_grad)
  return

def test_divide():
  # EWiseDiv: Op to element-wise divide two nodes
  print("=========EWiseDiv=======")
  # compute
  v1 = ndl.Tensor([3, 2, 1], dtype="float32")
  v2 = ndl.Tensor([2, 2, 2], dtype="float32")
  v3 = ndl.divide(v1, v2)
  print(v3)
  # gradient
  out_grad = ndl.Tensor([1, 1, 1], dtype="float32")
  v3_grad = v3.op.gradient(out_grad, v3)
  v1_3_grad = v3_grad[0]
  v2_3_grad = v3_grad[1]
  print("v1_3_grad.inputs = ", v1_3_grad.inputs, " -> gradient = ", v1_3_grad)
  print("v2_3_grad.inputs = ", v2_3_grad.inputs, " -> gradient = ", v2_3_grad)
  return

def test_divide_scalar():
  print("=========DivScalar=======")
  # compute
  v1 = ndl.Tensor([3, 2], dtype="float32")
  # v2 = v1/2
  v2 = ndl.divide_scalar(v1, 2)
  print(v2)
  # gradient
  out_grad = ndl.Tensor([1, 1], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ", v2_grad.inputs, " -> gradient = ", v2_grad)
  return

def test_transpose():
  print("=========Transpose=======")
  # compute
  v1 = ndl.Tensor([[1,2], [2,3], [3,4]], dtype="float32")
  print(v1)
  v2 = ndl.transpose(v1)
  # v2 = ndl.transpose(v1, [1, 0])
  print(v2)
  # gradient
  out_grad = ndl.Tensor([[1, 2, 3], [2, 3, 4]], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ", v2_grad.inputs)
  print(" -> gradient = ", v2_grad)
  return

def test_reshape():
  print("=========Reshape=======")
  # compute
  v1 = ndl.Tensor([[1,2], [2,3], [3,4]], dtype="float32")
  print(v1)
  v2 = ndl.reshape(v1, shape=(6,))
  print(v2)
  # gradient
  out_grad = ndl.Tensor([1, 2, 2, 3, 3, 4], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ", v2_grad.inputs)
  print(" -> gradient = ", v2_grad)
  return

def test_broadcast_to():
  print("=========BroadcastTo=======")
  # compute
  v1 = ndl.Tensor([1, 2, 3], dtype="float32")
  print(v1)
  # v2 = ndl.broadcast_to(v1, (3, 3))
  v2 = ndl.broadcast_to(v1, (4, 3))
  print(v2)
  # gradient
  out_grad = ndl.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ")
  print(v2_grad.inputs)
  print(" -> gradient = ")
  print(v2_grad)
  return

def test_summation():
  print("=========Summation=======")
  # compute
  v1 = ndl.Tensor([[1, 2, 3], [1, 2, 3]], dtype="float32")
  print(v1)
  v2 = ndl.summation(v1, 0)
  print(v2)
  # gradient
  out_grad = ndl.Tensor([2, 4, 6], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ")
  print(v2_grad.inputs)
  print(" -> gradient = ")
  print(v2_grad)
  return

def test_matmul():
  print("=========MatMul=======")
  # compute
  v1 = ndl.Tensor([[1, 0], [0, 1]], dtype="float32")
  v2 = ndl.Tensor([[4, 1], [2, 2]], dtype="float32")
  v3 = ndl.matmul(v1, v2)
  print(v3)
  # gradient
  out_grad = ndl.Tensor([[4, 1], [2, 2]], dtype="float32")
  v3_grad = v3.op.gradient(out_grad, v3)
  v1_3_grad = v3_grad[0]
  print("v1_3_grad.inputs = ")
  print(v1_3_grad.inputs)
  print(" -> gradient = ")
  print(v1_3_grad)
  v2_3_grad = v3_grad[1]
  print("v2_3_grad.inputs = ")
  print(v2_3_grad.inputs)
  print(" -> gradient = ")
  print(v2_3_grad)
  return

def test_negate():
  print("=========Negate=======")
  # compute
  v1 = ndl.Tensor([[1, -1]], dtype="float32")
  print(v1)
  v2 = ndl.negate(v1)
  print(v2)
  # gradient
  out_grad = ndl.Tensor([-1, 1], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ")
  print(v2_grad.inputs)
  print(" -> gradient = ")
  print(v2_grad)
  return

def test_log():
  print("=========Log=======")
  # compute
  v1 = ndl.Tensor([[1, 2]], dtype="float32")
  print(v1)
  v2 = ndl.log(v1)
  print(v2)
  # gradient
  out_grad = ndl.Tensor([0, 0.6931472], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ")
  print(v2_grad.inputs)
  print(" -> gradient = ")
  print(v2_grad)
  return

def test_exp():
  print("=========Exp=======")
  # compute
  v1 = ndl.Tensor([0, 1, 2], dtype="float32")
  print(v1)
  v2 = ndl.exp(v1)
  print(v2)
  # gradient
  out_grad = ndl.Tensor([1, 1, 1], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ")
  print(v2_grad.inputs)
  print(" -> gradient = ")
  print(v2_grad)
  return

def test_relu():
  print("=========ReLU=======")
  # compute
  v1 = ndl.Tensor([0, 1, -2, -1], dtype="float32")
  print(v1)
  v2 = ndl.relu(v1)
  print(v2)
  # gradient
  out_grad = ndl.Tensor([1, 1, 0, 0], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ")
  print(v2_grad.inputs)
  print(" -> gradient = ")
  print(v2_grad)
  return

def test_logsoftmax():
  print("=========LogSoftmax=======")
  # compute
  v1 = ndl.Tensor([0, 1, -1], dtype="float32")
  # v1 = ndl.Tensor([1000, 10000, 100], dtype="float32")
  print(v1)
  v2 = ndl.logsoftmax(v1)
  print(v2)
  # gradient
  out_grad = ndl.Tensor([1, 1, 1], dtype="float32")
  v2_grad = v2.op.gradient(out_grad, v2)
  print("v2_grad.inputs = ")
  print(v2_grad.inputs)
  print(" -> gradient = ")
  print(v2_grad)
  return

def test_my_solution():
  # test_power_scalar()
  # test_divide()
  # test_divide_scalar()
  # test_transpose()
  test_reshape()
  # test_broadcast_to()
  # test_summation()
  # test_matmul()
  # test_negate()
  # test_log()
  # test_exp()
  # test_relu()
  # test_logsoftmax()
  return

if __name__ == "__main__":
  # create_a_needle_tensor()
  # computational_graph()
  # executing_computation()
  # lazy_evaluation()
  # sum_loss()
  # reverse_mode_ad()
  test_my_solution()
