from tanka.predule import Variable


def test_rosenbrock():
    def rosenbrock(x0, x1):
        y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
        return y

    x0 = Variable(0.0)
    x1 = Variable(2.0)
    lr = 0.001
    iters = 10_0

    for _ in range(iters):
        print(x0, x1)
        y = rosenbrock(x0, x1)
        x0.zero_grad()
        x1.zero_grad()
        y.backward()

        x0.data -= lr * x0.grad.data
        x1.data -= lr * x1.grad.data


if __name__ == "__main__":
    test_rosenbrock()
