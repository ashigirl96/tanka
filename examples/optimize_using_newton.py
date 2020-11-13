from tanka.predule import Variable


def fn(x):
    y = x ** 4 - 2 * x ** 2
    return y


if __name__ == "__main__":
    x = Variable(2.0)
    iters = 10

    for i in range(iters):
        print(f"{i=} {x=}")

        y = fn(x)
        y.backward(create_graph=True)
        gx = x.grad
        x.zero_grad()

        gx.backward()
        g2x = x.grad
        x.zero_grad()

        x.data -= gx.data / g2x.data
