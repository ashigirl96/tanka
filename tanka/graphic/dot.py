from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from tanka.predule import Function, Variable


def _dot_var(v: Variable, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)
    return dot_var.format(id(v), name)


def _dot_fn(fn: Function):
    dot_fn = '{} [label="{}", color=lightblue, style=filled shape=box]\n'
    txt = dot_fn.format(id(fn), fn.__class__.__name__)

    dot_edge = "{} -> {}\n"
    for x in fn.inputs:
        txt += dot_edge.format(id(x), id(fn))
    for y in fn.outputs:
        txt += dot_edge.format(id(fn), id(y()))  # cause y is weakref
    return txt


def get_dot_graph(output: Variable, verbose=False):
    txt = ""
    fns: List[Function] = []
    seen_set = set()

    def add_fn(fn: Function):
        if fn not in seen_set:
            fns.append(fn)
            seen_set.add(fn)

    add_fn(output.creator_fn)
    txt += _dot_var(output, verbose)

    while fns:
        fn = fns.pop()
        txt += _dot_fn(fn)
        for x in fn.inputs:
            txt += _dot_var(x, verbose)

            if x.creator_fn is not None:
                add_fn(x.creator_fn)
    return "digraph g {\n" + txt + "}"


def plot_dot_graph(output: Variable, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    tanka_dir = Path.home() / ".tanka"
    tanka_dir.mkdir(exist_ok=True)
    graph_path = tanka_dir / "tmp_graph.dot"
    with open(graph_path, "w") as f:
        f.write(dot_graph)

    suffix = Path(to_file).suffix[1:]
    saved = tanka_dir / to_file
    cmd = f"dot {graph_path.as_posix()} -T {suffix} -o {saved.as_posix()}"
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    from jax import random

    key = random.PRNGKey(42)
    x = Variable(random.normal(key, (2, 3)), "x")
    print(_dot_var(x))
    print(_dot_var(x, verbose=True))

    x0 = Variable(random.normal(key, (2, 3)), "x0")
    x1 = Variable(random.normal(key, (2, 3)), "x1")
    y = x0 + x1
    txt = _dot_fn(y.creator_fn)
    print(txt)
