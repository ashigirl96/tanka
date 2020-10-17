from __future__ import annotations

import jax.numpy as np

from .predule import Function, Variable


def square(x: Variable) -> Variable:
    fn = Square()
    return fn(x)


def exp(x: Variable) -> Variable:
    fn = Exp()
    return fn(x)


class Square(Function):
    def forward(self, x: np.ndarray):
        return x ** 2

    def backward(self, gy: np.ndarray):
        x = self.input_.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: np.ndarray):
        return np.exp(x)

    def backward(self, gy: np.ndarray):
        x = self.input_.data
        gx = np.exp(x) * gy
        return gx
