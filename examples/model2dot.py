from tanka import Variable
from tanka.graphic import plot_dot_graph


def goldstein(x: Variable, y: Variable):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
        30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
    )
    return z


if __name__ == "__main__":
    x = Variable(1.0, name="x")
    y = Variable(1.0, name="y")
    z = goldstein(x, y)
    z.backward()
    z.name = "z"
    plot_dot_graph(z, verbose=False, to_file="goldstein.png")
