import jax.numpy as jnp
import matplotlib.pyplot as plt

import tanka.functions as F
from tanka.predule import Variable

if __name__ == "__main__":
    x = Variable(jnp.linspace(-7, 7, 200))
    y = F.sin(x)
    y.backward(create_graph=True)

    logs = [y.data.flatten()]

    for _ in range(3):
        logs.append(x.grad.data.flatten())
        gx = x.grad
        x.zero_grad()
        gx.backward(create_graph=True)

    labels = ["y=sin(x)", "y'", "y^{(2)}", "y^{(3)}"]
    for i, v in enumerate(logs):
        plt.plot(x.data, logs[i], label=labels[i])
    plt.legend(loc="lower right")
    plt.show()
