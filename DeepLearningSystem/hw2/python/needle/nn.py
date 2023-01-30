"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from needle import nn

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    """
    y = XW + b
    """
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True))
        # if bias:
        #     self.bias = init.kaiming_uniform(out_features, 1, requires_grad=True)
        #     self.bias = self.bias.reshape((1, out_features))
        #     self.bias = Parameter(self.bias)
        # else:
        #     self.bias = None
        self.use_bias = bias
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        if self.use_bias:
            self.bias = init.kaiming_uniform(out_features, 1)
            self.bias = self.bias.reshape((1, out_features))
            self.bias = Parameter(self.bias, device=device, dtype=dtype)
        # else:
        #     self.bias = None
        ### END YOUR SOLUTION

    # def forward(self, X: Tensor) -> Tensor:
    #     ### BEGIN YOUR SOLUTION
    #     linear = X @ self.weight
    #     if self.bias:
    #         return linear + self.bias.broadcast_to(linear.shape)
    #     else:
    #         return linear

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if len(X.shape) > len(self.weight.shape) and X.shape[-1] != self.weight.shape[0]:
            if len(X.shape) == 4 and X.shape[1]*X.shape[2]*X.shape[3] == self.weight.shape[0]:
                tform = nn.Flatten()
                linear = tform(X) @ self.weight
            else:
                import pdb;pdb.set_trace()
        else:
            linear = X @ self.weight
        if self.use_bias:
            return linear + self.bias.broadcast_to(linear.shape)
        else:
            return linear
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        exp_sum = ops.logsumexp(logits, axes=(1, )).sum()
        z_y_sum = (logits * init.one_hot(logits.shape[1], y)).sum()
        return (exp_sum - z_y_sum) / logits.shape[0]
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        w = init.ones(self.dim, requires_grad=True)
        self.weight = Parameter(w, device=device, dtype=dtype)
        b = init.zeros(self.dim, requires_grad=True)
        self.bias = Parameter(b, device=device, dtype=dtype)
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        mean = x.sum((0, )) / batch_size
        # NOTE array with shape (4, ) is considered as a row, so it can be brcsto (2, 4) and cannot be brcsto (4, 2)
        x_minus_mean = x - mean.broadcast_to(x.shape)
        var = (x_minus_mean ** 2).sum((0, )) / batch_size
        
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

            x_std = ((var + self.eps) ** 0.5).broadcast_to(x.shape)
            x_normed = x_minus_mean / x_std
            return x_normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        else:
            # NOTE no momentum here!
            x_normed = (x - self.running_mean) / (self.running_var + self.eps) ** 0.5
            # NOTE testing time also need self.weight and self.bias
            return x_normed * self.weight.broadcast_to(x.shape) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        w = init.ones(self.dim, requires_grad=True)
        self.weight = Parameter(w, device=device, dtype=dtype)
        b = init.zeros(self.dim, requires_grad=True)
        self.bias = Parameter(b, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature_size = x.shape[1]
        # NOTE need reshape, because (4, ) can brcsto (2, 4) but (4, ) cannot brcsto (4, 2)
        mean = x.sum(axes=(1, )).reshape((batch_size, 1)) / feature_size
        
        # NOTE need manual broadcast_to!!!
        x_minus_mean = x - mean.broadcast_to(x.shape)
        x_std = ((x_minus_mean ** 2).sum(axes=(1, )).reshape((batch_size, 1)) / feature_size + self.eps) ** 0.5
        # NOTE need manual broadcast_to!!!
        normed = x_minus_mean / x_std.broadcast_to(x.shape)
        
        return self.weight.broadcast_to(x.shape) * normed + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            prob = init.randb(*x.shape, p=1 - self.p)
            res = ops.multiply(x, prob) / (1 - self.p)
        else:
            res = x
        
        return res
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



