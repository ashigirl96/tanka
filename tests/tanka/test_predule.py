import jax.numpy as np
import pytest
from jax import grad
from numpy import testing

import tanka.functions as F
from tanka.numerical import numerical_grad
from tanka.predule import Variable


def ses(x):
    f1 = np.square(x)
    f2 = np.exp(f1)
    f3 = np.square(f2)
    return f3


def test_numerical_grad():
    data = np.array(2.0)
    x = Variable(data)
    result = numerical_grad(F.square, x)
    expect = grad(np.square)(data)
    # 数値誤差によて、decimalを小さくする必要がある
    testing.assert_almost_equal(result, expect, decimal=2)


def test_variable():
    with pytest.raises(TypeError):
        Variable(1)


def test_variable_backward():
    data = np.array(0.5)
    x = Variable(data)
    y = F.square(F.exp(F.square(x)))
    y.backward()

    dydx = grad(ses)(data)
    testing.assert_almost_equal(x.grad, dydx)
