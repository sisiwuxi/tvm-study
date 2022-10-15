"""
The main goal of this lecture is to create an accelerated ndarray library. 
As a result, we do not need to deal with needle.Tensor for now and will 
focus on backend_ndarray's implementation.

After we build up this array library, we can then use it to power backend 
array computations in needle.
"""
import sys
sys.path.append("./python")
from needle import backend_ndarray as nd
# import numpy as np

def create_cuda_ndarray():
    # Creating a CUDA NDArray
    x = nd.NDArray([1, 2, 3], device=nd.cpu())
    print("nd.cpu=", x)
    # We can create a CUDA tensor from the data by specifying a device keyword.
    # x = nd.NDArray([1, 2, 3], device=nd.cuda())
    # print("nd.ppu=", x, x.numpy, x.device)
    y = x + 1
    # trace
    # NDArray.__add__
    # NDArray.ewise_or_scalar
    # ndarray_backend_cpu.cc::ScalarAdd
    print(x.device.from_numpy)
    # NDArray.numpy
    # ndarray_backend_cpu.cc::to_numpy
    print("y=", y, y.numpy(), y.device)

    y = x + x
    print("y=", y, y.numpy(), y.device)
    return

def transformation_as_strided_computation():
    x = nd.NDArray([0,1,2,3,4,5], device=nd.cpu_numpy())
    print(x.numpy())
    y = nd.NDArray.make(shape=(3, 2, 2), strides=(2, 1, 0), device=x.device, handle=x._handle, offset=0)
    print(y.numpy())

    x = nd.NDArray([1, 2, 3, 4], device=nd.cpu_numpy())
    print(x.numpy())
    y = nd.NDArray.make(shape=(2, 2), strides=(2, 1), device=x.device, handle=x._handle, offset=0)
    print(y.numpy())

    z = nd.NDArray.make(shape=(2, 1), strides=(2, 1), device=x.device, handle=x._handle, offset=1)
    print(z.numpy())
    return

def cuda_operator():
    # NDArray.__truediv__
    # NDArray.ewise_or_scalar
    # ndarray_backend_cuda.cu::ScalarDiv
    # ndarray_backend_cuda.cu::ScalarDivKernel
    x = nd.NDArray([1,2,3], device=nd.cuda())
    y = nd.NDArray([2,3,4], device=nd.cuda())
    # z = x + x
    # z = x * x
    # z = x * 2
    # z = x / y
    # z = x / 2
    z = x ** 3
    # print("z = ", z, z.numpy(), z.device)
    print(x,y,z)
    return

def needle_tensor():
    # NDArray.__truediv__
    # NDArray.ewise_or_scalar
    # ndarray_backend_cpu.cc::ScalarDiv
    import needle as ndl
    x = ndl.Tensor([1,2,3], device=ndl.cpu(), dtype="float32")
    y = ndl.Tensor([2,3,5], device=ndl.cpu(), dtype="float32")
    # z = x + y
    # z = x * y
    # z = x / y
    z = x ** 2
    print("z = ", z, z.device, type(z.cached_data))
    return

if __name__ == "__main__":
    # create_cuda_ndarray()
    # transformation_as_strided_computation()
    cuda_operator()
    needle_tensor()