from typing import Tuple

import jax.numpy as jnp
from chex import Array

from ..predule import Function, Variable, VariableNum

Shape = Tuple[int, ...]


class Matmul(Function):
    def forward(self, *xs: Array) -> Array:
        return jnp.matmul(xs[0], xs[1])

    def backward(self, gy: Array) -> Tuple[Variable, Variable]:
        x = self.inputs[0]
        w = self.inputs[1]
        gx = matmul(gy, w.T)
        gw = matmul(x.T, gy)
        return gx, gw


def matmul(x: VariableNum, w: VariableNum) -> Variable:
    return Matmul()(x, w)
