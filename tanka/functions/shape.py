import jax.numpy as jnp
from chex import Array

from ..predule import Function, Shape, Variable, VariableNum, as_variable
from ..utility import reshape_sum_backward
from ..utility import sum_to as _sum_to


class Reshape(Function):
    shape: Shape

    def __init__(self, new_shape: Shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape

    def forward(self, x: Array) -> Array:
        self.shape = x.shape
        return jnp.reshape(x, self.new_shape)

    def backward(self, gy: Array) -> Variable:
        gx = reshape(gy, self.shape)
        return gx


class Transpose(Function):
    def forward(self, x: Array) -> Array:
        return x.transpose()

    def backward(self, gy: Array) -> Variable:
        return transpose(gy)


class BroadcastTo(Function):
    shape: Shape
    x_shape: Shape

    def __init__(self, shape: Shape):
        self.shape = shape

    def forward(self, x: Array) -> Array:
        self.x_shape = x.shape
        y = jnp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: Array) -> Variable:
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x: VariableNum, shape: Shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    shape: Shape
    x_shape: Shape

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: Array) -> Array:
        self.x_shape = x.shape
        y = _sum_to(x, self.shape)
        return y

    def backward(self, gy: Array) -> Variable:
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x: VariableNum, shape: Shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class Sum(Function):
    x_shape: Shape

    def __init__(self, axis, keepdims):
        super(Sum, self).__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: Array) -> Array:
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Array) -> Variable:
        gy = reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def reshape(x: VariableNum, shape: Shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x: VariableNum):
    return Transpose()(x)


def sum_(x: VariableNum, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)
