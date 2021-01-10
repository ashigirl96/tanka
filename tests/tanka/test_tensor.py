import jax.numpy as jnp

import tanka.functions as F
from tanka import Variable
from tanka.testing import equal_array


def test_tensor_sin():
    x = Variable(jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32))
    y = F.sin(x)
    print(y)


def test_sum():
    x = Variable(jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32))
    c = Variable(jnp.array([[10, 20, 30], [40, 50, 60]], dtype=jnp.float32))
    t = x + c
    y: jnp.ndarray = F.sum(t)
    assert y.data == jnp.array(231, jnp.float32)


def test_reshape():
    raw_x = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    raw_y = jnp.reshape(raw_x, (6,))
    x = Variable(raw_x)
    y = x.reshape((6,))
    equal_array(y.data, raw_y)

    _ = y.backward(retain_grad=True)
    gx = x.grad
    print(y.grad)
    print(gx)


def test_transpose():
    raw_x = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    raw_y = raw_x.T

    x = Variable(raw_x)
    y = x.T
    equal_array(y.data, raw_y)

    y.name = "fuga"
    y.backward(retain_grad=True)
    gx = x.grad
    print(y.grad)
    print(gx)


if __name__ == "__main__":
    # test_tensor_sin()
    # test_sum()
    # test_reshape()
    test_transpose()
