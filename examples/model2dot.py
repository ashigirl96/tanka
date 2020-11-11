from tanka import Variable
from tanka.graphic import plot_dot_graph

x0 = Variable(1.0, "x0")
x1 = Variable(1.0, "x1")
y = x0 + x1
y.name = "y"

plot_dot_graph(y, verbose=False)
