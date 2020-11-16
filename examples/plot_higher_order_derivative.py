import tanka.functions as F
from tanka import Variable
from tanka.graphic import plot_dot_graph

if __name__ == "__main__":
    x = Variable(1.0, name="x")
    y = F.tanh(x)
    y.name = "y"
    for i in range(5):
        x.zero_grad()
        y.backward(create_graph=True)
        y = x.grad
        y.name = f"gx{i + 1}"
    plot_dot_graph(y, verbose=False, to_file="higher_order_grad5.png")
