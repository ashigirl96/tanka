import chex as c
import jax.numpy as np
from jax import grad, random


def grad_grad_grad_exp():
    print(np.exp(0.5))
    print(grad(grad(grad(np.exp)))(0.5))


def sigmoid(x: c.Numeric):
    return 0.5 * (np.tanh(x * 0.5) + 1)


def predict(w: c.Numeric, b: c.Numeric, inputs: c.Numeric):
    return sigmoid(inputs.dot(w) + b)


def loss(w, b, inputs):
    preds = predict(w, b, inputs)
    label_props = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_props))


def loss2(params, inputs):
    preds = predict(params["w"], params["b"], inputs)
    label_props = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_props))


if __name__ == "__main__":
    key = random.PRNGKey(0)
    grad_grad_grad_exp()

    inputs = np.array(
        [[0.52, 1.12, 0.77], [0.88, -1.08, 0.15], [0.52, 0.06, -1.30], [0.74, -2.49, 1.39]]
    )
    targets = np.array([True, True, False, True])
    key, w_key, b_key = random.split(key, 3)
    w = random.normal(w_key, (3,))
    b = random.normal(b_key, (4,))

    w_grad = grad(loss, argnums=0)(w, b, inputs)
    print(f"{w_grad=}")

    w_grad = grad(loss)(w, b, inputs)
    print(f"{w_grad=}")

    b_grad = grad(loss, 1)(w, b, inputs)
    print(f"{b_grad=}")

    w_grad, _b_grad = grad(loss, (0, 1))(w, b, inputs)
    print(f"{w_grad=}\n{b_grad=}")

    # dictで渡せば、微分の方向考えなくて済む
    print(f"{grad(loss2)({'w': w, 'b': b}, inputs)=}")
