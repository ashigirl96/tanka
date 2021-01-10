import jax.numpy as jnp
import numpy as np


def equal_array(x: jnp.ndarray, y: jnp.ndarray):
    np.testing.assert_array_equal(x, y)


def equal(x: jnp.ndarray, y: float):
    return x == jnp.array(y)
