# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
import numpy as np
from tvm import relax
import torchvision
import torch
from tvm import meta_schedule as ms
import pickle as pkl
import IPython

def code2html(code):
    """Helper function to use pygments to turn the code string into highlighted html."""
    import pygments
    from pygments.lexers import Python3Lexer
    from pygments.formatters import HtmlFormatter
    formatter = HtmlFormatter()
    html = pygments.highlight(code, Python3Lexer(), formatter)
    return "<style>%s</style>%s\n" % (formatter.get_style_defs(".highlight"), html)

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(
        A: T.Buffer[(128, 128), "float32"],
        B: T.Buffer[(128, 128), "float32"],
        C: T.Buffer[(128, 128), "float32"],
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

def schedule_mm(sch: tvm.tir.Schedule, jfactor=4):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_0, j_1 = sch.split(loop=j, factors=[None, jfactor])
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch

def stochastic_schedule_mm(sch: tvm.tir.Schedule):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_factors = sch.sample_perfect_tile(loop=j, n=2)
    # print(type(j_factors[0]))
    j_0, j_1 = sch.split(loop=j, factors=j_factors)
    sch.reorder(i, j_0, k, j_1)
    t = sch.decompose_reduction(block_C, k)
    # print(t)
    return sch

def run_MyModule():
  dtype = "float32"
  a_np = np.random.rand(128, 128).astype(dtype)
  b_np = np.random.rand(128, 128).astype(dtype)
  c_mm = a_np @ b_np

  a_nd = tvm.nd.array(a_np)
  b_nd = tvm.nd.array(b_np)
  c_nd = tvm.nd.empty((128, 128), dtype="float32")
  lib = tvm.build(MyModule, target="llvm")
  f_timer_before = lib.time_evaluator("main", tvm.cpu())
  print("Time cost of MyModule: %.3f ms" % (f_timer_before(a_nd, b_nd, c_nd).mean * 1000))

def run_schedule_mm():
  dtype = "float32"
  a_np = np.random.rand(128, 128).astype(dtype)
  b_np = np.random.rand(128, 128).astype(dtype)
  c_mm = a_np @ b_np

  a_nd = tvm.nd.array(a_np)
  b_nd = tvm.nd.array(b_np)
  c_nd = tvm.nd.empty((128, 128), dtype="float32")

  sch = tvm.tir.Schedule(MyModule)
  sch = schedule_mm(sch)
  # IPython.display.HTML(code2html(sch.mod.script()))
  # print(sch.mod.script())
  lib = tvm.build(sch.mod, target="llvm")
  f_timer_after = lib.time_evaluator("main", tvm.cpu())
  print("Time cost of MyModule=>schedule_mm: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
  # print(sch.trace)

def run_stochastic_schedule_mm():
  dtype = "float32"
  a_np = np.random.rand(128, 128).astype(dtype)
  b_np = np.random.rand(128, 128).astype(dtype)
  c_mm = a_np @ b_np

  a_nd = tvm.nd.array(a_np)
  b_nd = tvm.nd.array(b_np)
  c_nd = tvm.nd.empty((128, 128), dtype="float32")

  sch = tvm.tir.Schedule(MyModule)
  sch = stochastic_schedule_mm(sch)
  # IPython.display.HTML(code2html(sch.mod.script()))
  # print(sch.mod.script())
  lib = tvm.build(sch.mod, target="llvm")
  f_timer_after = lib.time_evaluator("main", tvm.cpu())
  print("Time cost of MyModule=>stochastic_schedule_mm: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
  # print(sch.trace)

def run_random_search():
  dtype = "float32"
  a_np = np.random.rand(128, 128).astype(dtype)
  b_np = np.random.rand(128, 128).astype(dtype)
  c_mm = a_np @ b_np

  a_nd = tvm.nd.array(a_np)
  b_nd = tvm.nd.array(b_np)
  c_nd = tvm.nd.empty((128, 128), dtype="float32")
  def random_search(mod: tvm.IRModule, num_trials=5):
    best_result = None
    best_sch = None
    for i in range(num_trials):
        sch = stochastic_schedule_mm(tvm.tir.Schedule(mod))
        lib = tvm.build(sch.mod, target="llvm")
        f_timer_after = lib.time_evaluator("main", tvm.cpu())
        result = f_timer_after(a_nd, b_nd, c_nd).mean
        print("=====Attempt %d, time-cost: %.3f ms====" % (i, result * 1000))
        # book keep the best result so far
        if best_result is None or result < best_result:
            best_result = result
            best_sch = sch    
    return best_sch
  sch = random_search(MyModule)
  print(sch.trace)

def run_tune_with_space():
  dtype = "float32"
  a_np = np.random.rand(128, 128).astype(dtype)
  b_np = np.random.rand(128, 128).astype(dtype)
  c_mm = a_np @ b_np

  a_nd = tvm.nd.array(a_np)
  b_nd = tvm.nd.array(b_np)
  c_nd = tvm.nd.empty((128, 128), dtype="float32")
  sch_tuned = ms.tune_tir(
      mod=MyModule,
      target="llvm --num-cores=1",
      config=ms.TuneConfig(
        max_trials_global=64,
        num_trials_per_iter=64,
      ),
      space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
      work_dir="./tune_tmp",
      task_name="main"
  )
  lib = tvm.build(sch_tuned.mod, target="llvm")
  f_timer_after = lib.time_evaluator("main", tvm.cpu())
  print("Time cost of MyModule after tuning with space: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
  # print(sch_tuned.trace)
  # print(sch_tuned.mod.script())


def run_tune_without_space():
  dtype = "float32"
  a_np = np.random.rand(128, 128).astype(dtype)
  b_np = np.random.rand(128, 128).astype(dtype)
  c_mm = a_np @ b_np

  a_nd = tvm.nd.array(a_np)
  b_nd = tvm.nd.array(b_np)
  c_nd = tvm.nd.empty((128, 128), dtype="float32")
  sch_tuned = ms.tune_tir(
      mod=MyModule,
      target="llvm --num-cores=1",
      config=ms.TuneConfig(
        max_trials_global=64,
        num_trials_per_iter=64,
      ),
      # space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
      work_dir="./tune_tmp",
      task_name="main"
  )
  lib = tvm.build(sch_tuned.mod, target="llvm")
  f_timer_after = lib.time_evaluator("main", tvm.cpu())
  print("Time cost of MyModule after tuning without space: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
  # print(sch_tuned.trace)
  # print(sch_tuned.mod.script())

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


@tvm.script.ir_module
class MyModuleMixture: 
    @T.prim_func
    def linear0(X: T.Buffer[(1, 784), "float32"], 
                W: T.Buffer[(128, 784), "float32"], 
                B: T.Buffer[(128,), "float32"], 
                Z: T.Buffer[(1, 128), "float32"]):
        T.func_attr({"global_symbol": "linear0", "tir.noalias": True})
        Y = T.alloc_buffer((1, 128), "float32")
        for i, j, k in T.grid(1, 128, 784):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
    
        for i, j in T.grid(1, 128):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] =  Y[vi, vj] + B[vj]

    @R.function
    def main(x: Tensor((1, 784), "float32"), 
             w0: Tensor((128, 784), "float32"), 
             b0: Tensor((128,), "float32"), 
             w1: Tensor((10, 128), "float32"), 
             b1: Tensor((10,), "float32")):
        with R.dataflow():
            lv0 = R.call_tir(linear0, (x, w0, b0), (1, 128), dtype="float32")
            lv1 = R.call_tir("env.relu", (lv0,), (1, 128), dtype="float32")
            out = R.call_tir("env.linear", (lv1, w1, b1), (1, 10), dtype="float32")
            R.output(out)
        return out

@tvm.register_func("env.linear", override=True)
def torch_linear(x: tvm.nd.NDArray, 
                 w: tvm.nd.NDArray, 
                 b: tvm.nd.NDArray, 
                 out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)
    torch.mm(x_torch, w_torch.T, out=out_torch)
    torch.add(out_torch, b_torch, out=out_torch)

@tvm.register_func("env.relu", override=True)
def lnumpy_relu(x: tvm.nd.NDArray, 
                out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    out_torch = torch.from_dlpack(out)
    torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)


def run_model_MyModuleMixture(img, class_names):
  data_nd = tvm.nd.array(img.reshape(1, 784))
  mlp_params = pkl.load(open("/home/sisi/D/study/mlc.ai/fasionmnist_mlp_params.pkl", "rb"))
  nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

  MyModuleWithParams = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
  # IPython.display.HTML(code2html(MyModuleWithParams.script()))
  # print(MyModuleWithParams.script())

  ex = relax.vm.build(MyModuleWithParams, target="llvm")
  vm = relax.VirtualMachine(ex, tvm.cpu())

  nd_res = vm["main"](data_nd)

  pred_kind = np.argmax(nd_res.numpy(), axis=1)
  print("MyModuleWithParams Prediction:", class_names[pred_kind[0]])
  ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=100)
  print("MyModuleWithParams time-cost: %g ms" % (ftimer(data_nd).mean * 1000))
  return



def run_model_tune_linear0(img, class_names):
  data_nd = tvm.nd.array(img.reshape(1, 784))
  mlp_params = pkl.load(open("/home/sisi/D/study/mlc.ai/fasionmnist_mlp_params.pkl", "rb"))
  nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}

  mod_linear = tvm.IRModule.from_expr(MyModuleMixture["linear0"].with_attr("global_symbol", "main"))
  # IPython.display.HTML(code2html(mod_linear.script()))
  print(mod_linear.script())

  sch_tuned_linear = ms.tune_tir(
      mod=mod_linear,
      target="llvm --num-cores=1",
      config=ms.TuneConfig(
        max_trials_global=64,
        num_trials_per_iter=64,
      ),
      work_dir="./tune_tmp",
      task_name="main",
  )
  MyModuleWithParams2 = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
  new_func = sch_tuned_linear.mod["main"].with_attr("global_symbol", "linear0")
  gv = MyModuleWithParams2.get_global_var("linear0")
  MyModuleWithParams2.update_func(gv, new_func)
  print(sch_tuned_linear.trace)
  # IPython.display.HTML(code2html(MyModuleWithParams2.script()))
  print(MyModuleWithParams2.script())

  ex = relax.vm.build(MyModuleWithParams2, target="llvm")
  vm = relax.VirtualMachine(ex, tvm.cpu())
  nd_res = vm["main"](data_nd)
  pred_kind = np.argmax(nd_res.numpy(), axis=1)
  print("MyModuleWithParams2 Prediction:", class_names[pred_kind[0]])
  ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=50)
  print("MyModuleWithParams2 time-cost: %g ms" % (ftimer(data_nd).mean * 1000))
  return

if __name__ == '__main__':
  run_MyModule() # 3.534 ms
  run_schedule_mm() # 1.785 ms
  run_stochastic_schedule_mm() # 1.890 ms
  run_random_search() # 1.243 ms
  run_tune_with_space() # 0.101 ms
  run_tune_without_space() # 0.097 ms
  img, class_names = Load_the_Dataset()
  run_model_MyModuleMixture(img, class_names) # 0.276116 ms
  run_model_tune_linear0(img, class_names) # 0.0998983 ms
