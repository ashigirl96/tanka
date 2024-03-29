from typing import Tuple

import jax.numpy as jnp

from ..predule import Function, Variable


class Sin(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = jnp.sin(xs[0])
        return y

    def backward(self, *gys: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        x = self.inputs[0]
        gx = gys[0] * cos(x)
        return gx


class Cos(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = jnp.cos(xs[0])
        return y

    def backward(self, *gys: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        x = self.inputs[0]
        gx = gys[0] * (-sin(x))
        return gx


class Tanh(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = jnp.tanh(xs[0])
        return y

    def backward(self, gy: jnp.ndarray) -> Variable:
        y = self.outputs[0]()
        gx = (1.0 - y * y) * gy
        return gx


def sin(x: Variable):
    return Sin()(x)


def cos(x: Variable):
    return Cos()(x)


def tanh(x: Variable):
    return Tanh()(x)


if __name__ == "__main__":
    x = Variable(0.0)
    y = sin(x)
    y.backward()
    g1x = x.grad
    g1x.backward()
    g2x = g1x.grad
    print(y)
    print(g1x)
    print(g2x)
