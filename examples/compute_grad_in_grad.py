from tanka import Variable
from tanka.testing import equal


def fn(x):
    return x ** 2


def fn2(dydx, y):
    return dydx ** 3 + y


if __name__ == "__main__":
    x = Variable(2.0)
    y = fn(x)
    y.backward(create_graph=True)
    dydx = x.grad
    x.zero_grad()
    z = fn2(dydx, y)
    z.backward()
    dzdx = x.grad
    assert equal(dzdx.data, 100.0)
