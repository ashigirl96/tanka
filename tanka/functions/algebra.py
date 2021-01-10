from typing import Tuple

import jax.numpy as jnp
from chex import Array

from ..predule import Function, Variable, VariableNum, as_array


class Square(Function):
    def forward(self, x: Array):
        return x ** 2

    def backward(self, gy: Array):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: Array):
        return jnp.exp(x)

    def backward(self, gy: Array):
        x = self.inputs[0]
        gx = jnp.exp(x.data) * gy
        return gx


class Neg(Function):
    def forward(self, x: Array) -> Array:
        return -x

    def backward(self, gy: Array) -> Tuple[Array, ...]:
        return -gy


class Sub(Function):
    def forward(self, *xs: Array) -> Array:
        y = xs[0] - xs[1]
        return y

    def backward(self, gy: Array) -> Tuple[Array, Array]:
        return gy, -gy


class Add(Function):
    def forward(self, *xs: Array) -> Array:
        y = xs[0] + xs[1]
        return y

    def backward(self, gy: Array) -> Tuple[Array, Array]:
        return gy, gy


class Div(Function):
    def forward(self, *xs: Array) -> Array:
        y = xs[0] / xs[1]
        return y

    def backward(self, gy: Array) -> Tuple[Array, Array]:
        x0, x1 = self.inputs[0], self.inputs[1]
        gy0 = gy / x1
        gy1 = gy * (-x0 / x1 ** 2)
        return gy0, gy1


class Mul(Function):
    def forward(self, *xs: Array) -> Array:
        y = xs[0] * xs[1]
        return y

    def backward(self, gy: Array) -> Tuple[Array, Array]:
        x0, x1 = self.inputs[0], self.inputs[1]
        return gy * x1, gy * x0


class Pow(Function):
    def __init__(self, c: float):
        self.c = c

    def forward(self, x: Array):
        return x ** self.c

    def backward(self, gy: Array):
        x = self.inputs[0]
        gx = (self.c * x ** (self.c - 1)) * gy
        return gx


def neg(x: Variable) -> Variable:
    return Neg()(x)


def sub(x0: Variable, x1: VariableNum) -> Variable:
    x1 = as_array(x1)
    return Sub()(x0, x1)


def add(x0: Variable, x1: VariableNum) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)


def div(x0: Variable, x1: VariableNum) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: VariableNum) -> Variable:
    x1 = as_array(x1)
    return Div()(x1, x0)


def mul(x0: Variable, x1: VariableNum) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


def pow_(x: Variable, c: float) -> Variable:
    return Pow(c)(x)


def square(x: VariableNum) -> Variable:
    x = as_array(x)
    return Square()(x)


def exp(x: VariableNum) -> Variable:
    x = as_array(x)
    return Exp()(x)
