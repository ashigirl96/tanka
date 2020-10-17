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

    def backward(self) -> None:
        fns = [self.creator_fn]
        while fns:
            fn = fns.pop()
            if fn is None:
                continue
            x, y = fn.input_, fn.output
            x.grad = fn.backward(y.grad)

            if x.creator_fn is not None:
                fns.append(x.creator_fn)


class Function(ABC):
    def __call__(self, input_: Variable) -> Variable:
        y = self.forward(input_.data)
        output = Variable(y)

        self.input_ = input_
        self.output = output
        output.set_creator(self)
        return output

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, gy: np.ndarray) -> np.ndarray:
        pass
