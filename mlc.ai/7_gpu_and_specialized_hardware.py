# This is needed for deferring annotation parsing in TVMScript
from __future__ import annotations
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
from tvm import relax
import numpy as np


def vector_add():
  @tvm.script.ir_module
  class MyModuleVecAdd:
    @T.prim_func
    def main(A: T.Buffer[(1024,), "float32"], 
              B: T.Buffer[(1024,), "float32"], 
              C: T.Buffer[(1024,), "float32"]) -> None:
      T.func_attr({"global_symbol": "main", "tir.noalias": True})
      for i in T.grid(1024):
        with T.block("C"):
          vi = T.axis.remap("S", [i])
          C[vi] = A[vi] + B[vi]

  sch = tvm.tir.Schedule(MyModuleVecAdd)
  # sch.mod.show()
  block_C = sch.get_block("C")
  i, = sch.get_loops(block=block_C)
  i0, i1 = sch.split(i, [None, 128])
  # sch.mod.show()

  sch.bind(i0, "blockIdx.x")
  sch.bind(i1, "threadIdx.x")
  # sch.mod.show()

  rt_mod = tvm.build(sch.mod, target="cuda -arch=sm_35")

  A_np = np.random.uniform(size=(1024,)).astype("float32")
  B_np = np.random.uniform(size=(1024,)).astype("float32")
  A_nd = tvm.nd.array(A_np, tvm.cuda(0))
  B_nd = tvm.nd.array(B_np, tvm.cuda(0))
  C_nd = tvm.nd.array(np.zeros((1024,), dtype="float32"), tvm.cuda(0))

  rt_mod["main"](A_nd, B_nd, C_nd)
  print(A_nd)
  print(B_nd)
  print(C_nd)
  np.testing.assert_allclose(C_nd.numpy(), A_np + B_np)
  return

def window_sum():
  @tvm.script.ir_module
  class MyModuleWindowSum:
    @T.prim_func
    def main(A: T.Buffer[(1027,), "float32"], 
              B: T.Buffer[(1024,), "float32"]) -> None:
      T.func_attr({"global_symbol": "main", "tir.noalias": True})
      for i in T.grid(1024):
        with T.block("C"):
          vi = T.axis.remap("S", [i])
          B[vi] = A[vi] + A[vi + 1] + A[vi + 2]
  sch = tvm.tir.Schedule(MyModuleWindowSum)
  nthread = 128
  block_C = sch.get_block("C")
  i,  = sch.get_loops(block=block_C)
  i0, i1 = sch.split(i, [None, nthread])
  sch.bind(i0, "blockIdx.x")
  sch.bind(i1, "threadIdx.x")
  # sch.mod.show()

  A_shared = sch.cache_read(block_C, read_buffer_index=0, storage_scope="shared")
  sch.compute_at(A_shared, i1)
  # sch.mod.show()

  ax = sch.get_loops(A_shared)[-1]
  ax0, ax1 = sch.split(ax, [None, nthread])
  sch.bind(ax1, "threadIdx.x")
  # sch.mod.show()

  rt_mod = tvm.build(sch.mod, target="cuda -arch=sm_35")
  print(rt_mod.imported_modules[0].get_source())

  rt_mod = tvm.build(sch.mod, target="metal")
  print(rt_mod.imported_modules[0].get_source())
  return

def matrix_multiplication():
  @tvm.script.ir_module
  class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"], 
            B: T.Buffer[(1024, 1024), "float32"], 
            C: T.Buffer[(1024, 1024), "float32"]) -> None:
      T.func_attr({"global_symbol": "main", "tir.noalias": True})
      for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
          vi, vj, vk = T.axis.remap("SSR", [i, j, k])
          with T.init():
            C[vi, vj] = 0.0
          C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

  def blocking(sch, 
              tile_local_y, 
              tile_local_x, 
              tile_block_y, 
              tile_block_x,
              tile_k):
      block_C = sch.get_block("C")
      C_local = sch.cache_write(block_C, 0, "local")

      i, j, k = sch.get_loops(block=block_C)

      i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
      j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
      k0, k1 = sch.split(loop=k, factors=[None, tile_k])
      sch.unroll(k1)
      sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
      sch.reverse_compute_at(C_local, j1)

      sch.bind(i0, "blockIdx.y")
      sch.bind(j0, "blockIdx.x")

      sch.bind(i1, "threadIdx.y")
      sch.bind(j1, "threadIdx.x")
      sch.decompose_reduction(block_C, k0)
      return sch

  sch = tvm.tir.Schedule(MyModuleMatmul)
  # sch.mod.show()
  sch = blocking(sch, 8, 8, 8, 8, 4)
  # sch.mod.show()

  rt_mod = tvm.build(sch.mod, target="cuda -arch=sm_35")
  dev = tvm.cuda(0)
  A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
  B_np = np.random.uniform(size=(1024, 1024)).astype("float32")
  A_nd = tvm.nd.array(A_np, dev)
  B_nd = tvm.nd.array(B_np, dev)
  C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)

  num_flop = 2 * 1024 * 1024 * 1024
  evaluator = rt_mod.time_evaluator("main", dev, number=10)

  print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
  return

def matrix_multiplication_shared():
  @tvm.script.ir_module
  class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"], 
            B: T.Buffer[(1024, 1024), "float32"], 
            C: T.Buffer[(1024, 1024), "float32"]) -> None:
      T.func_attr({"global_symbol": "main", "tir.noalias": True})
      for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
          vi, vj, vk = T.axis.remap("SSR", [i, j, k])
          with T.init():
            C[vi, vj] = 0.0
          C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

  def cache_read_and_coop_fetch(sch, block, nthread, read_idx, read_loc):
      read_cache = sch.cache_read(block=block, read_buffer_index=read_idx, storage_scope="shared")
      sch.compute_at(block=read_cache, loop=read_loc)
      # vectorized cooperative fetch
      inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
      inner = sch.fuse(inner0, inner1)
      _, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
      sch.vectorize(vec)
      sch.bind(tx, "threadIdx.x")

  def blocking_with_shared(
      sch, 
      tile_local_y, 
      tile_local_x, 
      tile_block_y, 
      tile_block_x,
      tile_k):
      block_C = sch.get_block("C")
      C_local = sch.cache_write(block_C, 0, "local")

      i, j, k = sch.get_loops(block=block_C)

      i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
      j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
      k0, k1 = sch.split(loop=k, factors=[None, tile_k])

      sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
      sch.reverse_compute_at(C_local, j1)

      sch.bind(i0, "blockIdx.y")
      sch.bind(j0, "blockIdx.x")

      tx = sch.fuse(i1, j1)
      sch.bind(tx, "threadIdx.x")
      nthread = tile_block_y * tile_block_x
      cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)
      cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)    
      sch.decompose_reduction(block_C, k0)

      return sch

  sch = tvm.tir.Schedule(MyModuleMatmul)
  sch.mod.show()
  sch = blocking_with_shared(sch, 8, 8, 8, 8, 8)
  sch.mod.show()


  rt_mod = tvm.build(sch.mod, target="cuda -arch=sm_35")
  dev = tvm.cuda(0)
  A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
  B_np = np.random.uniform(size=(1024, 1024)).astype("float32")
  A_nd = tvm.nd.array(A_np, dev)
  B_nd = tvm.nd.array(B_np, dev)
  C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)

  num_flop = 2 * 1024 * 1024 * 1024
  evaluator = rt_mod.time_evaluator("main", dev, number=10)
  print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
  return

def automatic_program_optimization():
  @tvm.script.ir_module
  class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"], 
            B: T.Buffer[(1024, 1024), "float32"], 
            C: T.Buffer[(1024, 1024), "float32"]) -> None:
      T.func_attr({"global_symbol": "main", "tir.noalias": True})
      for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
          vi, vj, vk = T.axis.remap("SSR", [i, j, k])
          with T.init():
            C[vi, vj] = 0.0
          C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

  from tvm import meta_schedule as ms
  """
  gpuCount=1
  deviceId=0
  maxThreadsPerBlock:1024
  maxThreadsDim:1024
  maxGridSize:2147483647
  totalConstMem:65536
  clockRate:1176000
  integrated:0
  recommend deviceId=0
  """
  sch_tuned = ms.tune_tir(
      mod=MyModuleMatmul,
      # target="nvidia/tesla-p100",
      target="cuda -arch=sm_35 -max_threads_per_block=1024 -max_shared_memory_per_block=65536",
      config=ms.TuneConfig(
        max_trials_global=64,
        num_trials_per_iter=64,
      ),
      work_dir="./tune_tmp",
      task_name="main"
  )
  sch_tuned.mod.show()
  
  rt_mod = tvm.build(sch_tuned.mod, target="cuda -arch=sm_35 -max_threads_per_block=1024 -max_shared_memory_per_block=65536")
  dev = tvm.cuda(0)
  A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
  B_np = np.random.uniform(size=(1024, 1024)).astype("float32")
  A_nd = tvm.nd.array(A_np, dev)
  B_nd = tvm.nd.array(B_np, dev)
  C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)

  num_flop = 2 * 1024 * 1024 * 1024
  evaluator = rt_mod.time_evaluator("main", dev, number=10)
  print("MetaSchedule: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
  return

if __name__ == '__main__':
  # vector_add()
  # window_sum()
  # matrix_multiplication()
  # matrix_multiplication_shared()
  automatic_program_optimization()