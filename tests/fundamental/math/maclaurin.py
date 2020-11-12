import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, vmap


def fn(x):
    return (4 * x + 3) ** 3


if __name__ == "__main__":
    zero = jnp.linspace(0.0, 0.0)
    ones = jnp.ones_like(zero)
    g1fn = grad(fn)
    g2fn = grad(g1fn)
    g3fn = grad(g2fn)

    x = jnp.linspace(-0.5, 1.5)
    y1 = fn(x)
    y2 = fn(zero) + vmap(g1fn)(zero) * x
    # fn(0)で接してる
    y3 = fn(zero) + vmap(g1fn)(zero) * x + 0.5 * vmap(g2fn)(zero) * x ** 2
    y2t = fn(ones) + vmap(g1fn)(ones) * (x - ones)
    # fn(1)で接してる
    y3t = fn(ones) + vmap(g1fn)(ones) * (x - ones) + 0.5 * vmap(g2fn)(ones) * (x - ones) ** 2
    # y4 = fn(zero) + vmap(g1fn)(zero) * x + 0.5 * vmap(g2fn)(zero) * x ** 2 + (1. / 24) * vmap(g3fn)(zero) * x ** 3

    plt.plot(x, y1, label="0")
    plt.plot(x, y2, label="1m")
    plt.plot(x, y3, label="2m")
    plt.plot(x, y2t, label="1t")
    plt.plot(x, y3t, label="2t")
    plt.ylim(-25, 200)
    # plt.plot(x, y4, label='3')
    plt.legend()
    plt.show()
