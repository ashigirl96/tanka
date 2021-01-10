from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import jax.numpy as jnp

from .config import Config, using_config

Num = Union[int, float]
NumArray = Union[Num, jnp.ndarray]


class Variable:
    # 演算子の優先度. jnpだと100. npだと0.0
    __array_priority__ = 200

    def __init__(self, data: NumArray, name: str = None):
        if jnp.isscalar(data):
            data = jnp.array(data)
        if data is not None and not (isinstance(data, jnp.ndarray)):
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
        return transpose(self)

    @property
    def dtype(self):
        return self.data.dtype

    def reshape(self, shape: Shape):
        return reshape(self, shape)

    def sum(self, axis=None, keepdims=False):
        return sum_(self, axis, keepdims)

    def __neg__(self):
        return neg(self)

    def __add__(self, other: VarNum):
        return add(self, other)

    def __sub__(self, other: VarNum):
        return sub(self, other)

    def __radd__(self, other: VarNum):
        return add(self, other)

    def __rsub__(self, other: VarNum):
        return sub(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __mul__(self, other: VarNum):
        return mul(self, other)

    def __rmul__(self, other: VarNum):
        return mul(self, other)

    def __pow__(self, power, modulo=None):
        return pow_(self, power)


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
    def backward(self, *gys: jnp.ndarray) -> Variable:
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
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x: jnp.ndarray):
        return jnp.exp(x)

    def backward(self, gy: jnp.ndarray):
        x = self.inputs[0]
        gx = jnp.exp(x.data) * gy
        return gx


class Neg(Function):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return -x

    def backward(self, gy: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        return -gy


class Sub(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = xs[0] - xs[1]
        return y

    def backward(self, gy: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return gy, -gy


class Add(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = xs[0] + xs[1]
        return y

    def backward(self, gy: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return gy, gy


class Div(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = xs[0] / xs[1]
        return y

    def backward(self, gy: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x0, x1 = self.inputs[0], self.inputs[1]
        gy0 = gy / x1
        gy1 = gy * (-x0 / x1 ** 2)
        return gy0, gy1


class Mul(Function):
    def forward(self, *xs: jnp.ndarray) -> jnp.ndarray:
        y = xs[0] * xs[1]
        return y

    def backward(self, gy: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x0, x1 = self.inputs[0], self.inputs[1]
        return gy * x1, gy * x0


class Pow(Function):
    def __init__(self, c: float):
        self.c = c

    def forward(self, x: jnp.ndarray):
        return x ** self.c

    def backward(self, gy: jnp.ndarray):
        x = self.inputs[0]
        gx = (self.c * x ** (self.c - 1)) * gy
        return gx


Shape = Tuple[int, ...]


class Reshape(Function):
    shape: Shape

    def __init__(self, new_shape: Shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        self.shape = x.shape
        return jnp.reshape(x, self.new_shape)

    def backward(self, gy: jnp.ndarray) -> Variable:
        gx = reshape(gy, self.shape)
        return gx


class Transpose(Function):
    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.transpose()

    def backward(self, gy: jnp.ndarray) -> Variable:
        return transpose(gy)


class BroadcastTo(Function):
    shape: Shape
    x_shape: Shape

    def __init__(self, shape: Shape):
        self.shape = shape

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        self.x_shape = x.shape
        y = jnp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: jnp.ndarray) -> Variable:
        pass
        # gx = sum_to(gy, self.x_shape)
        # return gx


def broadcast_to(x: Variable, shape: Shape) -> Variable:
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    shape: Shape
    x_shape: Shape

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        pass
        # self.x_shape = x.shape
        # y = sum_to(x, self.shape)
        # return y

    def backward(self, gy: jnp.ndarray) -> Variable:
        gx = broadcast_to(gy, self.x_shape)
        return gx


class Sum(Function):
    x_shape: Shape

    def __init__(self, axis, keepdims):
        super(Sum, self).__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: jnp.ndarray) -> Variable:
        pass
        # gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        # gx = broadcast_to(gy, self.x_shape)
        # return gx


def neg(x: Variable) -> Variable:
    return Neg()(x)


def sub(x0: Variable, x1: VarNum) -> Variable:
    x1 = as_array(x1)
    return Sub()(x0, x1)


def add(x0: Variable, x1: VarNum) -> Variable:
    x1 = as_array(x1)
    return Add()(x0, x1)


def div(x0: Variable, x1: VarNum) -> Variable:
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: VarNum) -> Variable:
    x1 = as_array(x1)
    return Div()(x1, x0)


def mul(x0: Variable, x1: VarNum) -> Variable:
    x1 = as_array(x1)
    return Mul()(x0, x1)


def pow_(x: Variable, c: float) -> Variable:
    return Pow(c)(x)


def square(x: VarNum) -> Variable:
    x = as_array(x)
    return Square()(x)


def exp(x: VarNum) -> Variable:
    x = as_array(x)
    return Exp()(x)


def reshape(x: VarNum, shape: Shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x: VarNum):
    return Transpose()(x)


def sum_(x: VarNum, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)
