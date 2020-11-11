from tanka import Variable
from tanka.graphic import get_dot_graph, plot_dot_graph

x0 = Variable(1.0, "x0")
x1 = Variable(1.0, "x1")
y = x0 + x1
y.name = "y"

txt = get_dot_graph(y, verbose=False)
print(txt)

with open("sample.dot", "w") as o:
    o.write(txt)

plot_dot_graph(y, verbose=False)
