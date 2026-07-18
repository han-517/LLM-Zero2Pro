from __future__ import annotations

import math
from collections.abc import Callable


class Value:
    """一个最小标量自动微分节点，帮助观察计算图和链式法则。

    叶子节点的梯度和 PyTorch 参数梯度一样会在多次 ``backward`` 之间累加。
    中间节点只服务于一次反向传播，因此每次反传前都会清零，避免旧的链式
    法则贡献再次流入叶子。需要开始一个全新实验时，对输出调用 ``zero_grad``。
    """

    def __init__(
        self,
        data: float,
        children: tuple[Value, ...] = (),
        op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(children)
        self._op = op
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def _coerce(other: float | Value) -> Value:
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: float | Value) -> Value:
        other = self._coerce(other)
        output = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward
        return output

    def __radd__(self, other: float | Value) -> Value:
        return self + other

    def __mul__(self, other: float | Value) -> Value:
        other = self._coerce(other)
        output = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output

    def __rmul__(self, other: float | Value) -> Value:
        return self * other

    def __pow__(self, exponent: float) -> Value:
        if not isinstance(exponent, int | float):
            raise TypeError("Value 只支持数值指数")
        output = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * output.grad

        output._backward = _backward
        return output

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: float | Value) -> Value:
        return self + (-self._coerce(other))

    def __rsub__(self, other: float | Value) -> Value:
        return self._coerce(other) - self

    def __truediv__(self, other: float | Value) -> Value:
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other: float | Value) -> Value:
        return self._coerce(other) / self

    def exp(self) -> Value:
        result = math.exp(self.data)
        output = Value(result, (self,), "exp")

        def _backward() -> None:
            self.grad += result * output.grad

        output._backward = _backward
        return output

    def tanh(self) -> Value:
        result = math.tanh(self.data)
        output = Value(result, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - result**2) * output.grad

        output._backward = _backward
        return output

    def relu(self) -> Value:
        output = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (self.data > 0) * output.grad

        output._backward = _backward
        return output

    def _topological_order(self) -> list[Value]:
        order: list[Value] = []
        visited: set[Value] = set()

        def visit(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for child in node._prev:
                visit(child)
            order.append(node)

        visit(self)
        return order

    def zero_grad(self) -> None:
        """把当前输出可达的整张计算图梯度清零。"""

        for node in self._topological_order():
            node.grad = 0.0

    def backward(self) -> None:
        """从当前标量反向传播；中间梯度重算，叶子梯度累加。"""

        order = self._topological_order()
        for node in order:
            if node._prev:
                node.grad = 0.0

        if self._prev:
            self.grad = 1.0
        else:
            self.grad += 1.0
        for node in reversed(order):
            node._backward()
