"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
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
        return out_grad[0] + out_grad[1]


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
        return out_grad * self.scalar


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
        # return out_grad * self.scalar * (PowerScalar(self.scalar - 1)(lhs))
        return out_grad * self.scalar * array_api.power(lhs, (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes.
    true division of the inputs, element-wise (2 inputs)
    """

    def compute(self, a, b):
        # return array_api.divide(a, b)
        return a/b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # y'*(1/rhs)
        # import needle as ndl
        # # ones = array_api.ones(lhs.shape[0])
        # ones = array_api.ones(lhs.shape)
        # ones = ndl.Tensor(ones, dtype=lhs.dtype)
        # lhs_gradient = out_grad * EWiseDiv()(ones, rhs)
        lhs_gradient = out_grad/rhs
        # y'*-(u/v^2)
        # rhs_gradient = out_grad * (-1) * EWiseDiv()(lhs, PowerScalar(2)(rhs))
        rhs_gradient = out_grad * (-1) * lhs / rhs / rhs
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
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    """
    reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, axes - tuple)
    """
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # import numpy as np
        # a_len = len(a.shape)
        # if self.axes != None:
        #     axes1 = self.axes[0]
        #     axes2 = self.axes[1]
        #     if len(self.axes) != a_len:
        #         self.axes = np.arange(a_len)
        #         tmp = self.axes[axes1]
        #         self.axes[axes1] = self.axes[axes2]
        #         self.axes[axes2] = tmp
        # else:
        #     self.axes = np.arange(a_len)
        #     last = self.axes[-1]
        #     self.axes[-1] = self.axes[-2]
        #     self.axes[-2] = last
        # return array_api.transpose(a, self.axes)
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        return array_api.swapaxes(a, x, y)

    def gradient(self, out_grad, node):
        # lhs, = node.inputs
        # if self.axes is None:
        #     return array_api.transpose(out_grad)
        # # the original shape order: from 0 to len-1
        # axes = array_api.argsort(self.axes)
        # return Transpose(list(axes))(out_grad)
        if self.axes:
            x, y = self.axes[0], self.axes[1]
        else:
            x, y = -1, -2
        return transpose(out_grad, axes=(x, y))

def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    """
    gives a new shape to an array without changing its data (1 input, shape - tuple)
    """
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)
        # return a.numpy().reshape(self.shape)
        # return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        # return array_api.reshape(out_grad, lhs.shape)
        # return Reshape(lhs.shape)(out_grad)
        return reshape(out_grad, lhs.shape)


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
        shape = list(lhs.shape)
        axes = []
        shape = [1] * (len(self.shape) - len(shape)) + shape
        for i, s in enumerate(self.shape):
            if i >= len(shape) or s != shape[i]:
                axes.append(i)
        return reshape(summation(out_grad, tuple(axes)), lhs.shape)

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

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, = node.inputs

        # lhs_len = len(lhs.shape)
        # out_grad_len = len(out_grad.shape)
        # if lhs_len > out_grad_len:
        #     new_shape = array_api.ones(lhs_len, dtype=numpy.int8)
        #     for i in range(lhs_len):
        #         if i not in self.axes:
        #             new_shape[i] = lhs.shape[i]
        #     out_grad = Reshape(list(new_shape))(out_grad)
        
        # return BroadcastTo(lhs.shape)(out_grad)

        shape = lhs.shape
        shape_out = [1] * len(shape)
        if self.axes:
            s = set(self.axes)
        else:
            s = set(range(len(shape)))
        j = 0
        for i in range(len(shape)):
            if i not in s:
                shape_out[i] = out_grad.shape[j]
                j += 1
        # print(self.axes, out_grad.shape, shape_out, shape)
        result =  broadcast_to(reshape(out_grad, tuple(shape_out)), shape)
        return result


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    """
    matrix multiplication of the inputs (2 inputs)
    """
    def compute(self, a, b):
        # return array_api.matmul(a, b)
        return a @ b

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        grad_a = matmul(out_grad, transpose(rhs))
        grad_b = matmul(transpose(lhs), out_grad)
        if grad_a.shape != lhs.shape:
            length = len(grad_a.shape) - len(lhs.shape)
            grad_a = summation(grad_a, axes=tuple(range(length)))
        if grad_b.shape != rhs.shape:
            length = len(grad_b.shape) - len(rhs.shape)
            grad_b = summation(grad_b, axes=tuple(range(length)))
        return grad_a, grad_b
        ### END YOUR SOLUTION

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    """
    numerical negative, element-wise (1 input)
    """
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        # return Negate()(out_grad)
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        # import needle as ndl
        # ones = array_api.ones(lhs.shape)
        # ones = ndl.Tensor(ones, dtype=lhs.dtype)
        # return out_grad * EWiseDiv()(ones, lhs)
        return out_grad/lhs

def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        lhs, = node.inputs
        return out_grad * exp(lhs)


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    """
    relu(x) = max(0,x)
    """
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0].realize_cached_data()
        mask = Tensor(lhs > 0, requires_grad=False)
        return out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        exp_z = array_api.exp(Z - max_z)
        sum_z = array_api.sum(exp_z, axis=self.axes, keepdims=True)
        log_z = array_api.log(sum_z)
        log_sum_exp_z = log_z + max_z

        if self.axes:   
            out_shape = [size for i, size in enumerate(Z.shape) if i not in self.axes]
        else:
            out_shape = ()
        return array_api.resize(log_sum_exp_z, tuple(out_shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, = node.inputs
        if self.axes:
            shape = [1] * len(lhs.shape)
            s = set(self.axes)
            j = 0
            for i in range(len(shape)):
                if i not in s:
                    shape[i] = node.shape[j]
                    j += 1
            node_new = node.reshape(shape)
            grad_new = out_grad.reshape(shape)
        else:
            node_new = node
            grad_new = out_grad
        # print(node.shape, lhs.shape, node_new.shape, out_grad.shape)
        return grad_new * exp(lhs - node_new)

        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

