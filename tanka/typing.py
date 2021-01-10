import typing

from .predule import Function, Variable

Var = typing.TypeVar("Var", bound=Variable)
Fn = typing.TypeVar("Fn", bound=Function)
