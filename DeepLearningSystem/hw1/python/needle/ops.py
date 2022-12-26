"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


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
    """Op raise a tensor to an (integer) power.
    raise input to an integer (scalar) power
    """

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
    """Op to element-wise divide two nodes.
    true division of the inputs, element-wise (2 inputs)
    """

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
    """
    true division of the input by a scalar, element-wise (1 input, scalar - number)
    """
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
    reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, axes - tuple)
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
    """
    gives a new shape to an array without changing its data (1 input, shape - tuple)
    """
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # return array_api.reshape(a, self.shape)
        # return a.numpy().reshape(self.shape)
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        # return array_api.reshape(out_grad, lhs.shape)
        return Reshape(lhs.shape)(out_grad)

def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    """
    broadcast an array to a new shape (1 input, shape - tuple)
    """
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
    """
    sum of array elements over given axes (1 input, axes - tuple)
    """
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
    """
    matrix multiplication of the inputs (2 inputs)
    """
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
    """
    numerical negative, element-wise (1 input)
    """
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

