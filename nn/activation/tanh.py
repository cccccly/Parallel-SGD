import numpy as np
import collections
from nn.activation.abstract import AbsActivation
from nn.interface import IOperator


class Tanh(AbsActivation):

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

    def popleft_ref_input(self):
        self.__ref_input.popleft()

    def output_shape(self) -> [list, tuple, None]:
        return self.op_child.output_shape()

    def do_forward(self, x, training=True):
        self.__ref_input.append(np.tanh(x))
        if not training:
            return self.__ref_input.popleft()
        return self.__ref_input[-1]

    def do_backward(self, x, grad):
        return np.multiply(1 - np.square(self.__ref_input.popleft()), grad)
