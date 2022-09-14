"""Operator and gradient implementations."""
from ast import Pow
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
# compute is for value
# gradient is for value and computational graph both, so we can perform 
# gradient-of gradient computations
# out_grad is adjoint of output a.k.a \overline output
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return [out_grad[0] + out_grad[1]]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # for i in range(self.scalar):
        #     a = a*a
        # return a
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        # y' * s * x^{s-1}
        lhs, = node.inputs
        return out_grad * self.scalar * (PowerScalar(self.scalar - 1)(lhs))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.divide(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # y'*(1/rhs)
        import needle as ndl
        # ones = array_api.ones(lhs.shape[0])
        ones = array_api.ones(lhs.shape)
        ones = ndl.Tensor(ones, dtype=lhs.dtype)
        lhs_gradient = out_grad * EWiseDiv()(ones, rhs)
        # y'*-(u/v^2)
        rhs_gradient = out_grad * (-1) * EWiseDiv()(lhs, PowerScalar(2)(rhs))
        return lhs_gradient, rhs_gradient

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad * (1.0 / self.scalar)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    """
    reverse the order of two axes(axis1, axis2), defaults to the 
    last two axes(1 input, axes-tuple)
    """
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        import numpy as np
        a_len = len(a.shape)
        if self.axes != None:
            axes1 = self.axes[0]
            axes2 = self.axes[1]
            if len(self.axes) != a_len:
                self.axes = np.arange(a_len)
                tmp = self.axes[axes1]
                self.axes[axes1] = self.axes[axes2]
                self.axes[axes2] = tmp
        else:
            self.axes = np.arange(a_len)
            last = self.axes[-1]
            self.axes[-1] = self.axes[-2]
            self.axes[-2] = last
        return array_api.transpose(a, self.axes)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        if self.axes is None:
            return array_api.transpose(out_grad)
        # the original shape order: from 0 to len-1
        axes = array_api.argsort(self.axes)
        return Transpose(list(axes))(out_grad)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        # return array_api.reshape(out_grad, lhs.shape)
        return Reshape(lhs.shape)(out_grad)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        lhs_len = len(lhs.shape)
        out_grad_len = len(out_grad.shape)
        if lhs_len < out_grad_len:
            return Summation()(out_grad)
        if lhs_len == out_grad_len:
            for i in range(lhs_len):
                if lhs.shape[i] < out_grad.shape[i]:
                    axes = i
            ret = Summation(axes)(out_grad)
            ret = Reshape(lhs.shape)(ret)
            return ret


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        lhs_len = len(lhs.shape)
        out_grad_len = len(out_grad.shape)
        if lhs_len > out_grad_len:
            new_shape = array_api.ones(lhs_len, dtype=numpy.int8)
            for i in range(lhs_len):
                if i not in self.axes:
                    new_shape[i] = lhs.shape[i]
            out_grad = Reshape(list(new_shape))(out_grad)
        
        return BroadcastTo(lhs.shape)(out_grad)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        import numpy as np
        lhs, rhs = node.inputs
        lhs_len = len(lhs.shape)
        rhs_len = len(rhs.shape)
        out_len = len(out_grad.shape)
        axes = np.arange(len(lhs.shape))
        last = axes[-1]
        axes[-1] = axes[-2]
        axes[-2] = last
        axes = list(axes)
        lhs_t = Transpose(axes)(lhs)

        if lhs_len != rhs_len:
            axes = np.arange(rhs_len)
            last = axes[-1]
            axes[-1] = axes[-2]
            axes[-2] = last
            axes = list(axes)
        rhs_t = Transpose(axes)(rhs)
        lhs_grad = MatMul()(out_grad, rhs_t)
        rhs_grad = MatMul()(lhs_t, out_grad)
        lhs_grad_len = len(lhs_grad.shape)
        rhs_grad_len = len(rhs_grad.shape)
        if lhs_grad_len > lhs_len:
            for i in range(lhs_grad_len - lhs_len):
                lhs_grad = Summation(0)(lhs_grad)
        if rhs_grad_len > rhs_len:
            for i in range(rhs_grad_len - rhs_len):
                rhs_grad = Summation(0)(rhs_grad)
        
        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return Negate()(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        import needle as ndl
        ones = array_api.ones(lhs.shape)
        ones = ndl.Tensor(ones, dtype=lhs.dtype)
        return out_grad * EWiseDiv()(ones, lhs)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        return out_grad * Exp()(lhs)

def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        lhs_relu = ReLU()(lhs)
        lhs_1 = EWiseDiv()(lhs_relu, lhs)
        return out_grad * lhs_1


def relu(a):
    return ReLU()(a)


class LogSoftmax(TensorOp):
    def compute(self, Z):
        Z = Z - array_api.max(Z)
        z_exp = array_api.exp(Z)
        z_sum = array_api.sum(z_exp)
        softmax = array_api.divide(z_exp, z_sum)
        # softmax = softmax - array_api.max(softmax)
        log = array_api.log(softmax)
        return log

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        import needle as ndl
        ones = array_api.ones(node.shape[0])
        ones = ndl.Tensor(ones, dtype=node.dtype)
        reciprocal = EWiseDiv()(ones, node)
        # import pdb; pdb.set_trace()
        # s_i-s_i^2
        x1_s_grad = node - PowerScalar(2)(node)
        # -s_is_j
        return out_grad * reciprocal * x1_s_grad


def logsoftmax(a):
    return LogSoftmax()(a)


# additional helper functions
def full(
    shape, fill_value, *, rand={}, dtype="float32", device=None, requires_grad=False
):
    # numpy do not need device argument
    kwargs = {"device": device} if array_api is not numpy else {}
    device = device if device else cpu()

    if not rand or "dist" not in rand:
        arr = array_api.full(shape, fill_value, dtype=dtype, **kwargs)
    else:
        if rand["dist"] == "normal":
            arr = array_api.randn(
                shape, dtype, mean=rand["mean"], std=rand["std"], **kwargs
            )
        if rand["dist"] == "binomial":
            arr = array_api.randb(
                shape, dtype, ntrials=rand["trials"], p=rand["prob"], **kwargs
            )
        if rand["dist"] == "uniform":
            arr = array_api.randu(
                shape, dtype, low=rand["low"], high=rand["high"], **kwargs
            )

    return Tensor.make_const(arr, requires_grad=requires_grad)


def zeros(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    shape, *, mean=0.0, std=1.0, dtype="float32", device=None, requires_grad=False
):
    return full(
        shape,
        0,
        rand={"dist": "normal", "mean": mean, "std": std},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randb(shape, *, n=1, p=0.5, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "binomial", "trials": n, "prob": p},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randu(shape, *, low=0, high=1, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "uniform", "low": low, "high": high},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
