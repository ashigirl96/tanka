from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp


class Variable:
    def __init__(self, data: jnp.ndarray):
        if data is not None and not isinstance(data, jnp.ndarray):
            raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.grad: Optional[jnp.ndarray] = None
        self.creator_fn: Optional[Function] = None
        # 出力層に近いほど、大きくなる
        self.generation = 0

    def set_creator(self, fn: Function):
        self.creator_fn = fn
        self.generation = fn.generation + 1

    def backward(self) -> None:
        # 勾配開始の変数は勾配がないため、1を代入する
        if self.grad is None:
            self.grad = jnp.ones_like(self.data)

        fns: List[Function] = []
        seen_set = set()

        # 新しい世代の関数が後ろに追加するメソッド
        def add_fn(fn: Function):
            if fn not in seen_set:
                fns.append(fn)
                seen_set.add(fn)
                fns.sort(key=lambda x: x.generation)

        add_fn(self.creator_fn)

        while fns:
            fn = fns.pop()
            if fn is None:
                raise ValueError("There are invalid creator")
            # 一変数の場合
            # forwardのとき y <- fn.input(x)
            # x, y = fn.input, fn.output
            # x.grad = fn.backward(y.grad)
            # 多変数の場合
            gys = [output.grad for output in fn.outputs]
            gxs = fn.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(fn.inputs, gxs):
                if x.grad is None:
                    # 一回目のbackward
                    x.grad = gx
                else:
                    # 二回目移行のbackward
                    x.grad = x.grad + gx
                if x.creator_fn is not None:
                    add_fn(x.creator_fn)

    def zero_grad(self):
        self.grad = None
        # 一番最初の入力は関数によって生成された変数ではないので、そのままreturnする
        if self.creator_fn is None:
            return
        fns = [self.creator_fn]
        while fns:
            fn = fns.pop()
            for x in fn.inputs:
                x.grad = None
                if x.creator_fn is not None:
                    fns.append(x.creator_fn)

    def __repr__(self):
        return f"Variable({self.data})"


class Function(ABC):
    inputs: Tuple[Variable]
    outputs: List[Variable]
    generation: int

    def __call__(self, *inputs: Variable) -> Union[Variable, List[Variable]]:
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = [ys]
        outputs = [Variable(y) for y in ys]
        # 複数の入力だと、必ず世代が一致するわけではないので、最大のものを取る
        self.generation = max([input_.generation for input_ in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def backward(self, *gys: jnp.ndarray) -> Tuple[jnp.ndarray]:
        pass
