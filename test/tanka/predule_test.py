import jax.numpy as np
from jax import grad
from numpy import testing

from tanka.numerical import numerical_diff
from tanka.predule import Square, Variable


def test_numerical_diff():
    data = np.array(2.0)
    x = Variable(data)
    sq_fn = Square()
    result = numerical_diff(sq_fn, x)
    expect = grad(np.square)(data)
    testing.assert_almost_equal(result, expect, decimal=2)
