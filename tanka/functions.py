from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

from .predule import Function, Variable


def add(x0: Variable, x1: Variable) -> Variable:
    return Add()(x0, x1)


def square(x: Variable) -> Variable:
    return Square()(x)


def exp(x: Variable) -> Variable:
    return Exp()(x)


class DummyFunction(Function):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def backward(self, gy: jnp.ndarray):
        return gy


class Square(Function):
    def forward(self, x: jnp.ndarray):
        return x ** 2

    def backward(self, gy: jnp.ndarray):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: jnp.ndarray):
        return jnp.exp(x)

    def backward(self, gy: jnp.ndarray):
        x = self.inputs[0].data
        gx = jnp.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = xs[0] + xs[1]
        return y

    def backward(self, *gys: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return gys[0], gys[0]
