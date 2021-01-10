from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp
from chex import Array

from . import functions as F
from .config import Config, using_config

Num = Union[int, float]
NumArray = Union[Num, Array]
Shape = Tuple[int, ...]


class Variable:
    # 演算子の優先度. jnpだと100. npだと0.0
    __array_priority__ = 200

    def __init__(self, data: NumArray, name: str = None):
        if jnp.isscalar(data):
            data = jnp.array(data)
        if data is not None and not (isinstance(data, Array)):
            raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.name = name
        self.grad: Optional[Variable] = None
        self.creator_fn: Optional[Function] = None
        # 出力層に近いほど、大きくなる
        self.generation = 0

    def set_creator(self, fn: Function):
        self.creator_fn = fn
        self.generation = fn.generation + 1

    def backward(self, retain_grad=False, create_graph=False) -> None:
        # 勾配開始の変数は勾配がないため、1を代入する
        if self.grad is None:
            # self.grad = jnp.ones_like(self.data)
            self.grad = Variable(jnp.ones_like(self.data))

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
            gys = [output().grad for output in fn.outputs]
            print(f"{fn.outputs[0]().name=}")
            print(f"{fn.outputs[0]().grad=}")
            # create_graph = Falseのとき、__call__内でgeneration, inputs, outputsの保持をしなくなり逆伝搬を無効にする
            # 詳しくはP.235
            with using_config("enable_backprop", create_graph):
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
                    # 入力層以外の関数がfnsに追加されていく
                    if x.creator_fn is not None:
                        add_fn(x.creator_fn)
            # retain_grad = Falseのとき、中間の変数の勾配を無くす
            if not retain_grad:
                for output in fn.outputs:
                    output().grad = None

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Variable(\n{self.data}, {self.name})"

    def __len__(self):
        if self.ndim == 0:
            return 0
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return F.transpose(self)

    @property
    def dtype(self):
        return self.data.dtype

    def reshape(self, shape: Shape):
        return F.reshape(self, shape)

    def sum(self, axis=None, keepdims=False):
        return F.sum_(self, axis, keepdims)

    def __neg__(self):
        return F.neg(self)

    def __add__(self, other: VariableNum):
        return F.add(self, other)

    def __sub__(self, other: VariableNum):
        return F.sub(self, other)

    def __radd__(self, other: VariableNum):
        return F.add(self, other)

    def __rsub__(self, other: VariableNum):
        return F.sub(self, other)

    def __truediv__(self, other):
        return F.div(self, other)

    def __rtruediv__(self, other):
        return F.rdiv(self, other)

    def __mul__(self, other: VariableNum):
        return F.mul(self, other)

    def __rmul__(self, other: VariableNum):
        return F.mul(self, other)

    def __pow__(self, power, modulo=None):
        return F.pow_(self, power)


VariableNum = Union[Variable, Num, Array]


def as_variable(obj: VariableNum):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x: VariableNum):
    if jnp.isscalar(x):
        return jnp.array(x)
    return x


class Function(ABC):
    inputs: List[Variable]
    outputs: List[weakref.ReferenceType[Variable]]
    generation: int

    def __call__(self, *inputs: Union[Variable, Array]) -> Union[Variable, List[Variable]]:
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = [ys]
        outputs = [Variable(as_array(y)) for y in ys]
        # backprop=Falseのとき、すなわち順伝搬のときは関数に入力変数と出力変数の情報を保つ必要がないため、
        # 以下のように、入力と出力をインスタンス変数として持つ必要がなくなり、順伝搬した直後、参照カウントが0になり、
        # 変数が開放される
        if Config.enable_backprop:
            # 複数の入力だと、必ず世代が一致するわけではないので、最大のものを取る
            self.generation = max([input_.generation for input_ in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            # Variablesのcreate_fn(Function)とFunctionのoutputs(Variable)が循環参照になるので、
            # 弱参照を使う. TODO 弱参照の仕組み調べる
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    @abstractmethod
    def forward(self, *xs: Array) -> Array:
        pass

    @abstractmethod
    def backward(self, *gys: Array) -> Variable:
        pass


class DummyFunction(Function):
    def forward(self, x: Array) -> Array:
        return x

    def backward(self, gy: Array):
        return gy
