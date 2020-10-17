from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import jax.numpy as jnp


class Variable:
    def __init__(self, data: jnp.ndarray):
        if data is not None and not isinstance(data, jnp.ndarray):
            raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.grad: Optional[jnp.ndarray] = None
        self.creator_fn: Optional[Function] = None

    def set_creator(self, fn: Function):
        self.creator_fn = fn

    def backward(self) -> None:
        if self.grad is None:
            self.grad = jnp.ones_like(self.data)

        fns = [self.creator_fn]
        while fns:
            fn = fns.pop()
            if fn is None:
                raise ValueError("There are invalid creator")
            x, y = fn.input_, fn.output
            x.grad = fn.backward(y.grad)

            if x.creator_fn is not None:
                fns.append(x.creator_fn)


class Function(ABC):
    def __call__(self, input_: Variable) -> Variable:
        y = self.forward(input_.data)
        output = Variable(y)

        self.input_: Variable = input_
        self.output = output
        output.set_creator(self)
        return output

    @abstractmethod
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def backward(self, gy: jnp.ndarray) -> jnp.ndarray:
        pass
