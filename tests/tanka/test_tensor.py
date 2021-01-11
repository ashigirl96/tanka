import jax.numpy as jnp

import tanka.functions as F
import tanka.utility as utils
from tanka.predule import Variable
from tanka.testing import equal_array


def test_tensor_sin():
    x = Variable(jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32))
    y = F.sin(x)
    print(y)


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


def test_sum():
    raw_x = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    x = Variable(raw_x)
    y = F.sum_(x, axis=0)
    y.backward()
    gx = x.grad
    equal_array(gx.data, jnp.ones_like(raw_x))

    import jax

    x = Variable(jax.random.normal(jax.random.PRNGKey(42), (2, 3, 4, 5)))
    y = x.sum(keepdims=True)
    y.backward()
    gx = x.grad
    equal_array(gx.data, jnp.ones_like(x.data))


def test_sum_to():
    raw_x = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)
    raw_y = utils.sum_to(raw_x, (1, 3))
    expected = jnp.array([[5, 7, 9]], dtype=jnp.float32)
    equal_array(raw_y, expected)
    raw_y = utils.sum_to(raw_x, (2, 1))
    expected = jnp.array([[6], [15]], dtype=jnp.float32)
    equal_array(raw_y, expected)

    # ...: x = jnp.array([[1, 2, 3], [4, 5, 6]])
    # ...: y = sum_to(x, (2, 1))
    # ...: gy = jnp.ones_like(y)
    # ...: jnp.broadcast_to(gy, x.shape)

    raw_x = jnp.array([[1], [2]], dtype=jnp.float32)
    x = Variable(raw_x)
    y = F.broadcast_to(x, (2, 3))
    y.backward()
    gx = x.grad
    expected = jnp.array([[3], [3]], dtype=jnp.float32)
    print(f"{gx=}")
    equal_array(gx.data, expected)


def test_add():
    x0 = Variable(jnp.array([1, 2, 3]))
    x1 = Variable(jnp.array([10]))
    y = x0 + x1
    y.backward()
    gx0 = x0.grad
    gx1 = x1.grad
    equal_array(gx0.data, jnp.array([1, 1, 1]))
    equal_array(gx1.data, jnp.array([3]))
    print(y)


if __name__ == "__main__":
    # test_tensor_sin()
    # test_sum()
    # test_reshape()
    # test_transpose()
    # test_sum()
    test_add()
