import collections

import numpy as np

from nn.activation.abstract import AbsActivation
from nn.interface import IOperator


class ReLU(AbsActivation):

    def __init__(self, op: IOperator = None):
        super().__init__(op)
        self.__ref_input = collections.deque()

    @property
    def ref_input(self):
        return self.__ref_input

    def set_ref_input(self, ref_input):
        self.__ref_input.append(ref_input)

    def clear_ref_input(self):
        self.__ref_input.clear()

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        # put x into the end of deque
        self.__ref_input.append(x)
        # use the end of deque
        self.__ref_input[-1][self.__ref_input[-1] < 0] = 0
        res = self.__ref_input[-1]
        if not training:
            return self.__ref_input.popleft()
        return res

    def do_backward(self, x, grad):
        grad = np.multiply(grad, self.__ref_input[0] >= 0)
        self.__ref_input.popleft()
        return grad




class LeakReLU(AbsActivation):

    def __init__(self, leak_coefficient: float = 1e-2, op: IOperator = None):
        super().__init__(op)
        self.__leak_coef: float = leak_coefficient
        self.__mask: np.ndarray = np.ones(1)

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        self.__mask = x.copy()
        self.__mask[x > 0] = 1
        self.__mask[x <= 0] = self.__leak_coef
        return np.multiply(self.__mask, x)

    def do_backward(self, x, grad):
        return np.multiply(grad, self.__mask)

    def clear_unused(self):
        self.__mask = np.ones(shape=1)
