import jax.numpy as jnp

from tanka import Variable
from tanka.functions.geometry import sin


def test_sin():
    x = Variable(jnp.pi / 4)
    y = sin(x)
    y.backward()
    # d sin(x)/ dx = cos(x)
    assert y.data == x.grad
