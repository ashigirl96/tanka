import jax.numpy as jnp

from tanka.functions import add, square
from tanka.predule import Variable


def test_add():
    x0, x1 = Variable(jnp.array(2.0)), Variable(jnp.array(3.0))
    z = add(square(x0), square(x1))
    z.backward()
    print(z)
    print(x0.grad)
    print(x1.grad)
    assert x0.grad == jnp.array(4.0)
    assert x1.grad == jnp.array(6.0)


def test_add_same_variable():
    x0 = Variable(jnp.array(5.0))
    z = add(x0, add(x0, x0))
    z.backward()
    assert x0.grad == jnp.array(3.0)

    # 勾配を初期化しないと、x0の購買情報が残ってしまう
    x0.zero_grad()

    y = add(x0, add(x0, x0))
    y.backward()
    assert x0.grad == jnp.array(3.0)


def test_zero_grad():
    x = Variable(jnp.array(3.0))
    y = add(x, x)
    z = add(add(y, y), y)
    z.backward()
    assert y.grad == jnp.array(3.0)
    assert x.grad == jnp.array(6.0)

    z.zero_grad()
    assert z.grad is None
    assert y.grad is None
    assert x.grad is None
    z.backward()

    assert y.grad == jnp.array(3.0)
    assert x.grad == jnp.array(6.0)


def test_backward():
    x = Variable(jnp.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    assert y.data == jnp.array(32.0)
    assert x.grad == jnp.array(64.0)


if __name__ == "__main__":
    test_add_same_variable()
