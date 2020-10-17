from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax.numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.creator_fn: Optional[Function] = None

    def set_creator(self, fn: Function):
        self.creator_fn = fn


class Function(ABC):
    def __call__(self, input_: Variable) -> Variable:
        self.input_ = input_
        x = input_.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        return output

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, gy: np.ndarray) -> np.ndarray:
        pass


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
