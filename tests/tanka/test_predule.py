import jax.numpy as np
from jax import grad
from numpy import testing

from tanka.numerical import numerical_grad
from tanka.predule import Exp, Square, Variable


def test_numerical_grad():
    data = np.array(2.0)
    x = Variable(data)
    sq_fn = Square()
    result = numerical_grad(sq_fn, x)
    expect = grad(np.square)(data)
    # 数値誤差によて、decimalを小さくする必要がある
    testing.assert_almost_equal(result, expect, decimal=2)


def ses(x):
    f1 = np.square(x)
    f2 = np.exp(f1)
    f3 = np.square(f2)
    return f3


def test_backward():
    data = np.array(0.5)
    x = Variable(data)
    fn1 = Square()
    fn2 = Exp()
    fn3 = Square()
    x1 = fn1(x)
    x2 = fn2(x1)
    y = fn3(x2)
    y.grad = np.array(1.0)
    x2.grad = fn3.backward(y.grad)
    x1.grad = fn2.backward(x2.grad)
    x.grad = fn1.backward(x1.grad)

    dydx = grad(ses)(data)
    testing.assert_almost_equal(x.grad, dydx)
