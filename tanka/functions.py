from __future__ import annotations

import jax.numpy as jnp

from .predule import Function, Variable


def square(x: Variable) -> Variable:
    fn = Square()
    return fn(x)


def exp(x: Variable) -> Variable:
    fn = Exp()
    return fn(x)


class Square(Function):
    def forward(self, x: jnp.ndarray):
        return x ** 2

    def backward(self, gy: jnp.ndarray):
        x = self.input_.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: jnp.ndarray):
        return jnp.exp(x)

    def backward(self, gy: jnp.ndarray):
        x = self.input_.data
        gx = jnp.exp(x) * gy
        return gx
