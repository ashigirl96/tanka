import jax.numpy as jnp
from jax import grad
from numpy import testing

from tanka.numerical import numerical_grad
from tanka.predule import Variable, add, exp, square


def ses(x):
    f1 = jnp.square(x)
    f2 = jnp.exp(f1)
    f3 = jnp.square(f2)
    return f3


def test_numerical_grad():
    data = jnp.array(2.0)
    x = Variable(data)
    result = numerical_grad(square, x)
    expect = grad(jnp.square)(data)
    # 数値誤差によて、decimalを小さくする必要がある
    testing.assert_almost_equal(result, expect, decimal=2)


def test_variable_backward():
    data = jnp.array(0.5)
    x = Variable(data)
    y = square(exp(square(x)))
    y.backward()

    dydx = grad(ses)(data)
    testing.assert_almost_equal(x.grad, dydx)


def test_retain_graph():
    x0 = Variable(jnp.array(1.0))
    x1 = Variable(jnp.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()

    assert y.grad is None
    assert t.grad is None
    assert x0.grad == jnp.array(2.0)
    assert x1.grad == jnp.array(1.0)


def test_len():
    x = Variable(jnp.array(1.0))
    assert len(x) == 0
    from jax.random import PRNGKey, normal

    shape = (4, 5, 7)
    x = Variable(normal(PRNGKey(42), shape))
    assert len(x) == shape[0]


def test_shape():
    x = Variable(jnp.array(1.0))
    assert x.shape == ()

    from jax.random import PRNGKey, normal

    expect_shape = (3, 5, 7)
    x = Variable(normal(PRNGKey(42), expect_shape))
    assert x.shape == expect_shape


def test_ndim():
    assert Variable(jnp.array(1.0)).ndim == 0
    from jax.random import PRNGKey, normal

    shape = (4, 5, 7)
    assert Variable(normal(PRNGKey(42), shape)).ndim == len(shape)


def test_repr():
    from jax.random import PRNGKey, normal

    shape = (4, 5, 7)
    x = str(Variable(normal(PRNGKey(42), shape)))
    print(x)


def test_variable_mul():
    x0 = Variable(jnp.array(2.0))
    x1 = Variable(jnp.array(3.0))
    x2 = Variable(jnp.array(4.0))
    y = x0 * x1 + x2
    assert y.data == jnp.array(10.0)
    y.backward()
    assert x0.grad == jnp.array(3.0)
    assert x1.grad == jnp.array(2.0)
    assert x2.grad == jnp.array(1.0)


def test_variable_op_num():
    # neg
    assert (-Variable(1.0)).data == jnp.array(-1.0)
    # add
    assert (Variable(1.0) + 1.0).data == jnp.array(2.0)
    assert (1.0 + Variable(1.0)).data == jnp.array(2.0)
    # sub
    assert (Variable(1.0) - 1.0).data == jnp.array(0.0)
    assert (1.0 - Variable(1.0)).data == jnp.array(0.0)
    # mul
    assert (Variable(1.0) * 2.0).data == jnp.array(2.0)
    assert (2.0 * Variable(1.0)).data == jnp.array(2.0)
    # div
    assert (Variable(1.0) / 2.0).data == jnp.array(0.5)
    assert (2.0 / Variable(1.0)).data == jnp.array(2.0)
    # pow
    assert (Variable(2.0) ** 0).data == jnp.array(1.0)
    assert (Variable(2.0) ** 3).data == jnp.array(8.0)

    assert (1.0 + Variable(1.0) * 2.0).data == jnp.array(3.0)


def test_sphere():
    def sphere(x, y):
        z = x ** 2 + y ** 2
        return z

    x = Variable(1.0)
    y = Variable(1.0)
    z = sphere(x, y)
    z.backward()
    assert x.grad == y.grad


def test_matyas():
    def matyas(x, y):
        z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
        return z

    x = Variable(1.0)
    y = Variable(1.0)
    z = matyas(x, y)
    z.backward()
    testing.assert_almost_equal(x.grad, jnp.array(0.04), decimal=2)
    testing.assert_almost_equal(y.grad, jnp.array(0.04), decimal=2)


def test_variable_op_ndarray():
    x = Variable(jnp.array([1.0]))
    print(jnp.array([2.0]) + x)


if __name__ == "__main__":
    test_sphere()
