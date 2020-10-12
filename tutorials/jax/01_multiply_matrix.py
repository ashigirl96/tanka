import jax.numpy as jnp
import numpy as np
from jax import random

from tutorials.utility.timer import print_timer

if __name__ == "__main__":
    key = random.PRNGKey(0)
    x = random.normal(key, (10,))

    print(x)

    size = 3_000
    x = random.normal(key, (size, size), dtype=jnp.float32)
    np.random.normal(size=(size, size)).astype(np.float32)
    print_timer(lambda: jnp.dot(x, x.T).block_until_ready())
