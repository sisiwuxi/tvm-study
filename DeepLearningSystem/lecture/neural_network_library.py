from __future__ import print_function
import sys
sys.path.append("./python")
import needle as ndl
from util import *
# Now we are ready to design the interface of a high-level neural network library
#   data loader and preprocessing
#   initialization
#   compose the model
#   optimizer
#   loss function
# Let us start with the Module interface. We first introduce a parameter class 
# to indicate a Tensor is a trainable parameter.
class Parameter(ndl.Tensor):
  """parameter"""

def _get_params(value):
  if isinstance(value, Parameter):
    return [value]
  if isinstance(value, dict):
    params = []
    for k, v in value.items():
      params += _get_params(v)
    return params
  if isinstance(value, Module):
    return value.parameters()
  return []

class Module:
  def parameters(self):
    return _get_params(self.__dict__)

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)

# Now that we have the base Module interface, we can start to define different 
# kind of modules. Let us define a simple scale add module, that computes 
# y = x * s + b. The ScaleAdd is parameterized by s and b.

class ScaleAdd(Module):
  def __init__(self, init_s=1, init_b=0):
    self.s = Parameter([init_s], dtype="float32")
    self.b = Parameter([init_b], dtype="float32")
    return
  
  def forward(self, x):
    return x * self.s + self.b

# We allow a module to contain multiple submodules inside and compose them together
class MultiPathScaleAdd(Module):
  def __init__(self):
    self.path0 = ScaleAdd()
    self.path1 = ScaleAdd()
    return

  def forward(self, x):
    # y0 = x * s0 + b0 = x*1 + 0
    # y1 = x * s1 + b1 = x*1 + 0
    # y = y0 + y1 = x + x = 2*x
    return self.path0(x) + self.path1(x)

  def backward(self):
    return
#
# loss_function
#
class L2Loss(Module):
  def forward(self, x, y):
    z = x + (-1) * y
    self.h = x
    self.y = y
    self.z = z * z
    return z * z

#
# optimizer
#
# We are now ready to define the optimizer interface. There are two key 
# functions here:
#   reset_grad: reset the gradient fields of each the parameters
#   step: update the parameters
class Optimizer:
  def __init__(self, params):
    self.params = params

  def reset_grad(self):
    for p in self.params:
      p.grad = None
    
  def step(self):
    raise NotImplemented()

# stochastic gradient descent
class SGD(Optimizer):
  def __init__(self, params, lr):
    self.params = params
    self.lr = lr
    return

  def step(self):
    for w in self.params:
      w.data = w.data + (-self.lr) * w.grad
    return

class SGDWithMomentum(Optimizer):
  def __init__(self, params, lr):
    self.params = params
    self.lr = lr
    # Momentum
    # self.u = [ndl.Tensor([0], dtype="float32")]
    self.u = [ndl.zeros_like(p) for p in params]
    return

  def step(self):
    # import pdb;pdb.set_trace()
    for w in self.params:
      w.data = w.data + (-self.lr) * w.grad + self.u
    return