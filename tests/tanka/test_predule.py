import jax.numpy as jnp
import pytest
from jax import grad
from numpy import testing

import tanka.functions as F
from tanka.numerical import numerical_grad
from tanka.predule import Variable


def ses(x):
    f1 = jnp.square(x)
    f2 = jnp.exp(f1)
    f3 = jnp.square(f2)
    return f3


def test_numerical_grad():
    data = jnp.array(2.0)
    x = Variable(data)
    result = numerical_grad(F.square, x)
    expect = grad(jnp.square)(data)
    # 数値誤差によて、decimalを小さくする必要がある
    testing.assert_almost_equal(result, expect, decimal=2)


def test_variable():
    with pytest.raises(TypeError):
        Variable(1)


def test_variable_backward():
    data = jnp.array(0.5)
    x = Variable(data)
    y = F.square(F.exp(F.square(x)))
    y.backward()

    dydx = grad(ses)(data)
    testing.assert_almost_equal(x.grad, dydx)


def test_generations():
    from tanka.functions import DummyFunction

    generations = [2, 0, 1, 4, 2]
    fns = []
    for g in generations:
        fn = DummyFunction()
        fn.generation = g
        fns.append(fn)
    fns.sort(key=lambda x: x.generation)
    print(list(map(lambda x: x.generation, fns)))


if __name__ == "__main__":
    test_generations()
