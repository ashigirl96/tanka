import jax.numpy as jnp

from tanka.functions.geometry import sin
from tanka.predule import Variable


def test_sin():
    x = Variable(jnp.pi / 4)
    y = sin(x)
    y.backward()
    # d sin(x)/ dx = cos(x)
    assert y.data == x.grad.data
