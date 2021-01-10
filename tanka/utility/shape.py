from typing import Tuple

from chex import Array

Shape = Tuple[int, ...]


def sum_to(x: Array, shape: Shape) -> Array:
    y: Array
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple(i + lead for i, sx in enumerate(shape) if sx == 1)
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y
