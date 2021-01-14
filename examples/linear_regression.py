from dataclasses import dataclass

import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array

from tanka.functions.matrix import matmul
from tanka.predule import Variable


@dataclass
class LinearParams:
    w: Variable
    b: Variable


def predict(x: Variable, params: LinearParams) -> Variable:
    y = matmul(x, params.w) + params.b
    return y


def mse(logits: Variable, targets: Variable) -> Variable:
    batch_size = logits.shape[0]
    errors = (logits - targets) ** 2
    return errors.sum() / batch_size


def generate_dataset(rng: hk.PRNGSequence):
    while True:
        x = jax.random.normal(next(rng), (100, 1))
        y = 5 + 2 * x + jax.random.normal(next(rng), (100, 1))
        yield x, y


def plot_regression(x: Array, y: Array, params: LinearParams) -> None:
    min_x, max_x = x.min(keepdims=True), x.max(keepdims=True)
    predicted_min_y = predict(min_x, params).data
    predicted_max_y = predict(max_x, params).data
    plt.scatter(x, y)
    plt.plot(
        (min_x.squeeze(), max_x.squeeze()),
        (predicted_min_y.squeeze(), predicted_max_y.squeeze()),
        color="r",
    )


if __name__ == "__main__":
    total_steps = 100
    learning_rate = 0.1
    rng = hk.PRNGSequence(42)
    w = Variable(jnp.zeros((1, 1)))
    b = Variable(jnp.zeros(1))
    X = []
    Y = []

    for step, (x, y) in zip(range(total_steps), generate_dataset(rng)):
        X.extend(x)
        Y.extend(y)
        x = Variable(x)
        targets = Variable(y)
        logits = predict(x, params=LinearParams(w=w, b=b))
        loss = mse(logits, targets)
        w.zero_grad()
        b.zero_grad()
        loss.backward()

        w.data -= learning_rate * w.grad.data
        b.data -= learning_rate * b.grad.data
        print(f"Loss[{step}] = {loss.data}, w={w.data}, b={b.data}")

    x, y = next(generate_dataset(rng))
    plot_regression(x, y, LinearParams(w=w, b=b))
    plt.show()
