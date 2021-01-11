import jax.numpy as jnp

from tanka.functions.matrix import matmul
from tanka.predule import Variable
from tanka.testing import equal_array


def test_matmul():
    x = Variable(jnp.ones((2, 3)))
    w = Variable(jnp.ones((3, 4)))
    y = matmul(x, w)
    assert y.shape == (2, 4)

    y.backward()
    gx = x.grad
    gw = w.grad
    assert gx.shape == x.shape
    equal_array(gx.data, x.data * 4.0)
    assert gw.shape == w.shape
    equal_array(gw.data, w.data * 2.0)
