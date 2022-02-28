import tvm
import tvm.testing
from tvm import te
import numpy as np
import pdb

run_cuda = True
if run_cuda:
    # Change this target to the correct backend for you gpu. For example: cuda (NVIDIA GPUs),
    # rocm (Radeon GPUS), OpenCL (opencl).
    tgt_gpu = tvm.target.Target(target="cuda", host="llvm")

    # Recreate the schedule
    n = te.var("n")
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")
    C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
    print(type(C))

    s = te.create_schedule(C.op)

    bx, tx = s[C].split(C.op.axis[0], factor=64)

    ################################################################################
    # Finally we must bind the iteration axis bx and tx to threads in the GPU
    # compute grid. The naive schedule is not valid for GPUs, and these are
    # specific constructs that allow us to generate code that runs on a GPU.

    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))

    ######################################################################
    # Compilation
    # -----------
    # After we have finished specifying the schedule, we can compile it
    # into a TVM function. By default TVM compiles into a type-erased
    # function that can be directly called from the python side.
    #
    # In the following line, we use tvm.build to create a function.
    # The build function takes the schedule, the desired signature of the
    # function (including the inputs and outputs) as well as target language
    # we want to compile to.
    #
    # The result of compilation fadd is a GPU device function (if GPU is
    # involved) as well as a host wrapper that calls into the GPU
    # function. fadd is the generated host wrapper function, it contains
    # a reference to the generated device function internally.

    fadd = tvm.build(s, [A, B, C], target=tgt_gpu, name="myadd")

    ################################################################################
    # The compiled TVM function is exposes a concise C API that can be invoked from
    # any language.
    #
    # We provide a minimal array API in python to aid quick testing and prototyping.
    # The array API is based on the `DLPack <https://github.com/dmlc/dlpack>`_ standard.
    #
    # - We first create a GPU device.
    # - Then tvm.nd.array copies the data to the GPU.
    # - ``fadd`` runs the actual computation
    # - ``numpy()`` copies the GPU array back to the CPU (so we can verify correctness).
    #
    # Note that copying the data to and from the memory on the GPU is a required step.

    dev = tvm.device(tgt_gpu.kind.name, 0)

    n = 1024
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
    fadd(a, b, c)
    tvm.testing.assert_allclose(c.numpy(), a.numpy() + b.numpy())

    ################################################################################
    # Inspect the Generated GPU Code
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # You can inspect the generated code in TVM. The result of tvm.build is a TVM
    # Module. fadd is the host module that contains the host wrapper, it also
    # contains a device module for the CUDA (GPU) function.
    #
    # The following code fetches the device module and prints the content code.
    pdb.set_trace()
    if (
        tgt_gpu.kind.name == "cuda"
        or tgt_gpu.kind.name == "rocm"
        or tgt_gpu.kind.name.startswith("opencl")
    ):
        dev_module = fadd.imported_modules[0]
        print("-----GPU code-----")
        print(dev_module.get_source())
    else:
        print(fadd.get_source())