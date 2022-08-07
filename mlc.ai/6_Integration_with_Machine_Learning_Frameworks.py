# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations 
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
from tvm import relax
import numpy as np

import torch
import torch.nn as nn
from torch import fx
from torch.nn import functional as F

from tvm import te
import torch
import torchvision
import matplotlib.pyplot as plt

import pickle as pkl
from tvm import topi

def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")

def te_relu(A: te.Tensor) -> te.Tensor:
    return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")

def Tensor_Expression_for_TensorIR_Creation():
  A = te.placeholder((128, 128), name="A", dtype="float32")
  B = te.placeholder((128, 128), name="B", dtype="float32")
  print(type(A))
  C = te_matmul(A, B)
  # print(te.create_prim_func([A, B, C]))
  X1 = te.placeholder((10,), name="X1", dtype="float32")
  Y1 = te_relu(X1)
  # print(te.create_prim_func([X1, Y1]))
  X2 = te.placeholder((10, 20), name="X1", dtype="float32")
  Y2 = te_relu(X2)
  # print(te.create_prim_func([X2, Y2]))
  C = te_matmul(A, B)
  D = te_relu(C)
  # print(te.create_prim_func([A, B, D]))
  print(te.create_prim_func([A, B, C, D]))
  return

def Use_BlockBuilder_to_Create_an_IRModule():
  A = relax.Var("A", (128, 128), relax.DynTensorType(2, "float32"))
  B = relax.Var("B", (128, 128), relax.DynTensorType(2, "float32"))
  bb = relax.BlockBuilder()

  with bb.function("main"):
      with bb.dataflow():
          C = bb.emit_te(te_matmul, A, B)
          D = bb.emit_te(te_relu, C)
          R = bb.emit_output(D)
      bb.emit_func_output(R, params=[A, B])
  # print(type(C))
  # print(isinstance(C, relax.Var))
  MyModule = bb.get()
  # print(MyModule)
  return

def map_param(param: nn.Parameter):
  ndim = len(param.data.shape)
  return relax.const(
      param.data.cpu().numpy(), relax.DynTensorType(ndim, "float32")
  )

def fetch_attr(fx_mod, target: str):
  """Helper function to fetch an attr"""
  target_atoms = target.split('.')
  attr_itr = fx_mod
  for i, atom in enumerate(target_atoms):
      if not hasattr(attr_itr, atom):
          raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
      attr_itr = getattr(attr_itr, atom)
  return attr_itr

def from_fx(fx_mod, input_shapes, call_function_map, call_module_map):
  input_index = 0
  node_map = {}
  named_modules = dict(fx_mod.named_modules())

  bb = relax.BlockBuilder()

  fn_inputs = []
  fn_output = None
  with bb.function("main"):
      with bb.dataflow():
          for node in fx_mod.graph.nodes:
              if node.op == "placeholder":
                  # create input placeholder
                  shape = input_shapes[input_index]
                  input_index += 1 
                  input_var = relax.Var(
                      node.target, shape, relax.DynTensorType(len(shape), "float32")
                  )
                  fn_inputs.append(input_var)
                  node_map[node] = input_var
              elif node.op == "get_attr":
                  node_map[node] = map_param(fetch_attr(fx_mod, node.target))
              elif node.op == "call_function":
                  node_map[node] = call_function_map[node.target](bb, node_map, node)
              elif node.op == "call_module":
                  named_module = named_modules[node.target]
                  node_map[node] = call_module_map[type(named_module)](bb, node_map, node, named_module)
              elif node.op == "output":
                  output = node_map[node.args[0]]
                  assert fn_output is None
                  fn_output = bb.emit_output(output)
      # output and finalize the function
      bb.emit_func_output(output, fn_inputs)
  return bb.get()


def Import_Model_From_PyTorch():
  class MyModel(nn.Module):
    def __init__(self):
      super(MyModel, self).__init__()
      self.weight = nn.Parameter(torch.randn(128, 128))

    def forward(self, x):
      x = torch.matmul(x, self.weight)
      x = torch.relu(x)
      return x
  model = MyModel()
  fx_module = fx.symbolic_trace(model)
  # print(type(fx_module))
  # fx_module.graph.print_tabular()
  # print(fx_module.graph)

  def map_matmul(bb, node_map, node: fx.Node):
      A = node_map[node.args[0]]
      B = node_map[node.args[1]]
      return bb.emit_te(te_matmul, A, B)

  def map_relu(bb, node_map, node: fx.Node):
      A = node_map[node.args[0]]
      return bb.emit_te(te_relu, A)

  MyModule = from_fx(
      fx_module, 
      input_shapes = [(1, 128)], 
      call_function_map = {
        torch.matmul: map_matmul,
        torch.relu: map_relu, 
      },
      call_module_map={},
  )

  # MyModule.show()
  print(MyModule)

  return

class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.linear0 = nn.Linear(784, 128, bias=True)
    self.relu = nn.ReLU()
    self.linear1 = nn.Linear(128, 10, bias=True)

  def forward(self, x):
    x = self.linear0(x)
    x = self.relu(x)
    x = self.linear1(x)
    return x

def Load_the_Dataset():
  test_data = torchvision.datasets.FashionMNIST(
      root="data",
      train=False,
      download=True,
      transform=torchvision.transforms.ToTensor()
  )
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

  img, label = next(iter(test_loader))
  img = img.reshape(1, 28, 28).numpy()
  import matplotlib.pyplot as plt

  plt.figure()
  plt.imshow(img[0])
  plt.colorbar()
  plt.grid(False)
  # plt.show()
  plt.savefig("FashionMNIST.png")

  print("Golden Class:", class_names[label[0]])
  return img, class_names
  
def Coming_back_to_FashionMNIST_Example():
  # -------------- golden ------------ #
  img, class_names = Load_the_Dataset()

  # ------------- pickle torch ------------- #
  mlp_model = MLP()
  mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
  mlp_model.linear0.weight.data = torch.from_numpy(mlp_params["w0"])
  mlp_model.linear0.bias.data = torch.from_numpy(mlp_params["b0"])
  mlp_model.linear1.weight.data = torch.from_numpy(mlp_params["w1"])
  mlp_model.linear1.bias.data = torch.from_numpy(mlp_params["b1"])

  torch_res = mlp_model(torch.from_numpy(img.reshape(1, 784)))

  pred_kind = np.argmax(torch_res.detach().numpy(), axis=1)
  print("Torch Prediction:", class_names[pred_kind[0]])
  
  # ------------- topi prediction ------------- #
  def map_relu(bb, node_map, node: fx.Node):
      A = node_map[node.args[0]]
      return bb.emit_te(te_relu, A)

  def map_nn_linear(bb, node_map, node, nn_mod):
      x = node_map[node.args[0]]
      w = map_param(nn_mod.weight)
      if nn_mod.bias is not None:
          b = map_param(nn_mod.bias)
      y = bb.emit_te(topi.nn.dense, x, w)
      return bb.emit_te(topi.add, y, b)

  def map_nn_relu(bb, node_map, node, nn_mod):
      return map_relu(bb, node_map, node)

  MLPModule = from_fx(
      fx.symbolic_trace(mlp_model), 
      input_shapes = [(1, 784)], 
      call_function_map={
      },
      call_module_map={
          torch.nn.Linear: map_nn_linear,
          torch.nn.ReLU: map_nn_relu,
      },
  )

  # MLPModule.show()
  print(MLPModule)
  ex = relax.vm.build(MLPModule, target="llvm")
  vm = relax.VirtualMachine(ex, tvm.cpu())
  data_nd = tvm.nd.array(img.reshape(1, 784))

  nd_res = vm["main"](data_nd)

  pred_kind = np.argmax(nd_res.numpy(), axis=1)
  print("MLPModule Prediction:", class_names[pred_kind[0]])

  return

def Translating_into_High_level_Operators():
  # -------------- golden ------------ #
  img, class_names = Load_the_Dataset()

  # ------------- pickle torch ------------- #
  mlp_model = MLP()
  mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
  mlp_model.linear0.weight.data = torch.from_numpy(mlp_params["w0"])
  mlp_model.linear0.bias.data = torch.from_numpy(mlp_params["b0"])
  mlp_model.linear1.weight.data = torch.from_numpy(mlp_params["w1"])
  mlp_model.linear1.bias.data = torch.from_numpy(mlp_params["b1"])

  torch_res = mlp_model(torch.from_numpy(img.reshape(1, 784)))

  pred_kind = np.argmax(torch_res.detach().numpy(), axis=1)
  print("Torch Prediction:", class_names[pred_kind[0]])

  # ------------- relex prediction ------------- #
  def map_param(param: nn.Parameter):
    ndim = len(param.data.shape)
    return relax.const(
        param.data.cpu().numpy(), relax.DynTensorType(ndim, "float32")
    )

  def map_nn_relu_op(bb, node_map, node, nn_mod):
      A = node_map[node.args[0]]
      return bb.emit(relax.op.nn.relu(A))

  def map_nn_linear_op(bb, node_map, node, nn_mod):
      x = node_map[node.args[0]]
      w = map_param(nn_mod.weight)
      if nn_mod.bias is not None:
          b = map_param(nn_mod.bias)
      y = bb.emit(relax.op.nn.dense(x, w))
      return bb.emit(relax.op.add(y, b))

  MLPModuleHighLevel = from_fx(
      fx.symbolic_trace(mlp_model), 
      input_shapes = [(1, 784)], 
      call_function_map={
      },
      call_module_map={
          torch.nn.Linear: map_nn_linear_op,
          torch.nn.ReLU: map_nn_relu_op,
      },
  )

  # MLPModuleHighLevel.show()
  print(MLPModuleHighLevel)
  return

if __name__ == '__main__':
  # Tensor_Expression_for_TensorIR_Creation()
  # Use_BlockBuilder_to_Create_an_IRModule()
  # Import_Model_From_PyTorch()
  # Coming_back_to_FashionMNIST_Example()
  Translating_into_High_level_Operators()
