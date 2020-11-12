import jax.numpy as jnp


def equal(x: jnp.ndarray, y: float):
    return x == jnp.array(y)
