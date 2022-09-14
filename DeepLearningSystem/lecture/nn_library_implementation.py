from __future__ import print_function
import sys
sys.path.append("./python")
import needle as ndl
from util import *
import numpy as np
import neural_network_library as nnl

def needle_refresh():
  # Mutating the data field of a needle Tensor
  w = ndl.Tensor([1,2,3], dtype="float32")
  g = ndl.Tensor([1,1,1], dtype="float32")
  # By default, we create needle Tensors that sets requires_grad to be true. 
  # This will cause us to record the gradient graph by default.
  print(w.requires_grad)
  # stochastic gradient descent(sgd) style update
  grad = ndl.Tensor([1, 1, 1], dtype="float32")
  # learning rate
  lr = 0.1
  # w -> + -> + -> + -> + -> + -> w
  #      |    |    |    |    |
  #   (-lr)*g .    .    .    .
  # out of memory
  for i in range(5):
      w = w + (-lr) * grad
  print(w.op)
  print(w.inputs[0].op)
  print(w.inputs[0].inputs[0].op)
  print(w.data)
  # def data(self):
  #   return self.detach()
  # def detach(self):
  #   """Create a new tensor that shares the data but detaches from the graph."""
  #   return Tuple.make_const(self.realize_cached_data())
  # detach has the same content(deep copy) but do not contain computational graph
  print("-----------detach-----------")
  print(w.data.requires_grad)
  z = w.data
  print(z.op is None)
  print(z.inputs)
  print(z.cached_data is w.cached_data)
  print("-----------new_w-----------")
  # old: w = w + (-lr) * grad
  new_w = w.data + (-lr)*grad.data
  print(new_w)
  print(new_w.requires_grad)
  print(new_w.inputs)
  w.data = w.data + (-lr)*grad.data
  print(w)
  return

def numerical_stability():
  # Most computations in a deep learning model are executed using 32-bit floating
  #  point. We need to pay special attention to potential numerical problems. 
  # Softmax is one of the most commonly used operators in loss functions. 
  # Let z = softmax(x), then we have z(i) = exp(x(i))/sum_k(exp(x(k)))
  def softmax_naive(x):
      z = np.exp(x)
      return z / np.sum(z)

  # If we naively follow the formula to compute softmax, the result can be inaccurate.
  x = np.array([100, 100, 101], dtype="float32")
  p = softmax_naive(x)
  print("softmax_naive: ", p)

  # Passing a large number(that is greator than 0) to exp function can easily 
  # result in overflow. Note that the following invariance hold for any 
  # constant "c"
  # z(i) = exp(x(i))/sum_k(exp(x(k))) = exp(x(i)-c)/sum_k(exp(x(k)-c))
  # We can pick c=max(x(i)) so that all the inputs to the exp become smaller 
  # or equal to 0
  def softmax_stable(x):
    x = x - np.max(x)
    z = np.exp(x)
    return z / np.sum(z)
  
  x = np.array([1000, 10000, 100], dtype="float32")
  p = softmax_stable(x)
  print("softmax_stable: ", p)
  # Similar principles hold when we compute logsoftmax, or logsumexp operations.
  return

def nn_module_interface():
  print("-----nn.Module interface-----")
  w = nnl.Parameter([0, 1], dtype="float32")
  print(isinstance(w, nnl.Parameter))
  print("-----sadd default-----")
  sadd = nnl.ScaleAdd()
  print(sadd.parameters())
  print(sadd.parameters()[0] is sadd.s)
  print(sadd.parameters()[1] is sadd.b)
  x = ndl.Tensor([2], dtype="float32")
  print(sadd(x))
  print("-----sadd(2,1)-----")
  sadd = nnl.ScaleAdd(2, 1)
  print(sadd(x))
  mpath = nnl.MultiPathScaleAdd()
  print(mpath.parameters())
  print(mpath.parameters()[0] is mpath.path0.s)
  y = ndl.Tensor([1], dtype="float32")
  print(mpath(y))
  return

def loss_function():
  print("-----loss_function:ScaleAdd-----")
  x = ndl.Tensor([2], dtype="float32")
  y = ndl.Tensor([2], dtype="float32")
  sadd = nnl.ScaleAdd()
  # ((2*1+0) + (-1)*2) * ((2*1+0) + (-1)*2)
  #  2      1
  #   \    /
  #  EWiseMul    0       2
  #       \     /        |
  #       EWiseAdd  MulScalar(-1)
  #           \        /
  #            EWiseAdd
  #                   \
  #                    EWiseMul
  #                       |
  loss = nnl.L2Loss()(sadd(x), y)
  loss.backward()
  params = sadd.parameters()
  print(params[0].grad)

  print("-----loss_function:MultiPathScaleAdd-----")
  mpath = nnl.MultiPathScaleAdd()
  print(mpath.parameters())
  loss = nnl.L2Loss()(mpath(x), y)
  loss.backward()
  params = mpath.parameters()
  print(params[0].grad, mpath.path0.s.grad)
  print(params[1].grad, mpath.path0.b.grad)
  return

def designing_a_neural_network_library():
  """
  mix and match different modules together, they can effectively interact each other together
  hypothesis_class, loss_function, optimizer_method
  """
  x = ndl.Tensor([2], dtype="float32")
  y = ndl.Tensor([2], dtype="float32")
  # nn.Module interface
  model = nnl.MultiPathScaleAdd()
  print(model.parameters())
  # loss_function
  l2loss = nnl.L2Loss()
  # Optimizer
  opt = nnl.SGD(model.parameters(), lr=0.01)
  num_epoch = 10

  for epoch in range(num_epoch):
      opt.reset_grad()
      # hypothesis
      h = model(x)
      loss = l2loss(h, y)
      training_loss = loss.numpy()
      print(training_loss)
      loss.backward()
      opt.step()
  return

def fused_operator():
  x = ndl.Tensor([1], dtype="float32")
  z = ndl.ops.fused_add_scalars(x, 1, 2)
  v0 = z[0]
  v0.backward()
  print(v0.op, v0.grad)
  v1 = z[0] + z[1]*2
  v1.backward()
  print(v1.op, v1.grad)
  return

if __name__ == "__main__":
  # needle_refresh()
  # numerical_stability()
  # nn_module_interface()
  # loss_function()
  designing_a_neural_network_library()
  fused_operator()
  
