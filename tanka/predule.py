from abc import ABC, abstractmethod

import jax.numpy as np


class Variable:
    def __init__(self, data: np.ndarray):
        self.data = data


class Function(ABC):
    def __call__(self, input_: Variable) -> Variable:
        x = input_.data
        y = self.forward(x)
        output = Variable(y)
        return output

    @abstractmethod
    def forward(self, x: np.ndarray):
        pass


class Square(Function):
    def forward(self, x: np.ndarray):
        return x ** 2


class Exp(Function):
    def forward(self, x: np.ndarray):
        return np.exp(x)
