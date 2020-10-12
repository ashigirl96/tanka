import jax.numpy as np
from jax import grad, jit

from tutorials.utility.timer import print_timer


def sum_logistic(x: np.ndarray):
    return np.sum(1.0 / (1.0 / np.exp(-x)))


if __name__ == "__main__":
    x_small = np.arange(3.0)
    derivative_fn = grad(sum_logistic)
    print(x_small)
    print(derivative_fn(x_small))

    print_timer(lambda: grad(jit(grad(jit(sum_logistic))))(1.0))
    print_timer(lambda: grad(grad(sum_logistic))(1.0))
