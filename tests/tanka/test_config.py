import jax.numpy as jnp

import tanka.functions as F
from tanka.config import no_grad, using_config
from tanka.predule import Variable


def test_using_config():
    x = Variable(jnp.array(1.0))
    y = F.exp(x)
    assert hasattr(y.creator_fn, "inputs")
    assert hasattr(y.creator_fn, "generation")
    assert hasattr(y.creator_fn, "outputs")

    with using_config("enable_backprop", False):
        x = Variable(jnp.array(1.0))
        y = F.exp(x)
    assert not hasattr(y.creator_fn, "inputs")
    assert not hasattr(y.creator_fn, "generation")
    assert not hasattr(y.creator_fn, "outputs")


def test_no_grad():
    x = Variable(jnp.array(1.0))
    y = F.exp(x)
    assert hasattr(y.creator_fn, "inputs")
    assert hasattr(y.creator_fn, "generation")
    assert hasattr(y.creator_fn, "outputs")

    with no_grad():
        x = Variable(jnp.array(1.0))
        y = F.exp(x)
    assert not hasattr(y.creator_fn, "inputs")
    assert not hasattr(y.creator_fn, "generation")
    assert not hasattr(y.creator_fn, "outputs")
