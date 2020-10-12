import jax.numpy as np
from jax import jit, random

from tutorials.utility.timer import print_timer


def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * np.where(x > 0.0, x, alpha * np.exp(x) - alpha)


if __name__ == "__main__":
    key = random.PRNGKey(0)
    x = random.normal(key, (1_000_000,))
    print_timer(lambda: selu(x).block_until_ready())

    selu_jit = jit(selu)
    print_timer(lambda: selu_jit(x).block_until_ready())
