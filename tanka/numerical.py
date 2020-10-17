import jax.numpy as np

from .predule import Variable


def numerical_grad(fn, x: Variable, eps=1e-4) -> np.ndarray:
    y0 = fn(Variable(x.data - eps))
    y1 = fn(Variable(x.data + eps))
    return (y1.data - y0.data) / (2 * eps)
