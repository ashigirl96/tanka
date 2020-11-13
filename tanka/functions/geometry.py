from typing import Tuple

import jax.numpy as jnp

from tanka import Function, Variable


class Sin(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = jnp.sin(xs[0])
        return y

    def backward(self, *gys: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        x = self.inputs[0].data
        gx = gys[0] * jnp.cos(x)
        return gx


class Cos(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = jnp.cos(xs[0])
        return y

    def backward(self, *gys: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        x = self.inputs[0].data
        gx = gys[0] * (-jnp.sin(x))
        return gx


def sin(x: Variable):
    return Sin()(x)


def cos(x: Variable):
    return Cos()(x)


if __name__ == "__main__":
    x = Variable(0.0)
    print(sin(x))
