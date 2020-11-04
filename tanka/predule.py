from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp

Num = Union[int, float]
NumArray = Union[Num, jnp.ndarray]


class Config:
    enable_backprop = True


class Variable:
    def __init__(self, data: NumArray, name: str = None):
        if data is not None and not (isinstance(data, jnp.ndarray) or jnp.isscalar(data)):
            raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.name = name
        self.grad: Optional[jnp.ndarray] = None
        self.creator_fn: Optional[Function] = None
        # 出力層に近いほど、大きくなる
        self.generation = 0

    def set_creator(self, fn: Function):
        self.creator_fn = fn
        self.generation = fn.generation + 1

    def backward(self, retain_graph=False) -> None:
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
            gys = [output().grad for output in fn.outputs]
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
            if not retain_graph:
                for output in fn.outputs:
                    output().grad = None

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return f"Variable({self.data}, {self.name})"

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
    def dtype(self):
        return self.data.dtype

    def __add__(self, other: VarNum):
        return add(self, other)

    def __radd__(self, other: VarNum):
        return add(self, other)

    def __mul__(self, other: VarNum):
        return mul(self, other)

    def __rmul__(self, other: VarNum):
        return mul(self, other)


VarNum = Union[Variable, Num, jnp.ndarray]


def as_variable(obj: VarNum):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x: VarNum):
    if jnp.isscalar(x):
        return jnp.array(x)
    return x


class Function(ABC):
    inputs: List[Variable]
    outputs: List[weakref.ReferenceType[Variable]]
    generation: int

    def __call__(self, *inputs: Union[Variable, jnp.ndarray]) -> Union[Variable, List[Variable]]:
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
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        pass

    @abstractmethod
    def backward(self, *gys: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        pass


class DummyFunction(Function):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

    def backward(self, gy: jnp.ndarray):
        return gy


class Square(Function):
    def forward(self, x: jnp.ndarray):
        return x ** 2

    def backward(self, gy: jnp.ndarray):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: jnp.ndarray):
        return jnp.exp(x)

    def backward(self, gy: jnp.ndarray):
        x = self.inputs[0].data
        gx = jnp.exp(x) * gy
        return gx


class Add(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = xs[0] + xs[1]
        return y

    def backward(self, gy: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        print(f"{gy=}")
        return gy, gy


class Mul(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = xs[0] * xs[1]
        return y

    def backward(self, gy: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def add(x0: Variable, x1: VarNum) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)


def mul(x0: Variable, x1: VarNum) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


def square(x: VarNum) -> Variable:
    x = as_array(x)
    return Square()(x)


def exp(x: VarNum) -> Variable:
    x = as_array(x)
    return Exp()(x)
