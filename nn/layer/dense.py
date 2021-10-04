import collections

import numpy as np

from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.activation.interface import IActivation
from nn.value.trainable import Weights
import copy


class Dense(AbsLayer):

    def __init__(self, units, activation: IActivation = None, inputs: IOperator = None, max_batch_num: int = 1):
        super().__init__(inputs, activation, max_batch_num)
        self.__layer_units = units
        self.__w = Weights()
        self.__b = Weights()
        self.__w_queue = collections.deque([Weights() for i in range(self.max_batch_num)])
        self.__b_queue = collections.deque([Weights() for i in range(self.max_batch_num)])

    def output_shape(self) -> [list, tuple, None]:
        return [-1, self.__layer_units]

    @property
    def variables(self) -> list:
        res = []
        for w, b in zip(self.__w_queue, self.__b_queue):
            res.append((w, b))
        return res

    def initialize_parameters(self, x) -> None:
        high = np.sqrt(6 / (x.shape[1] + self.__layer_units))
        low = -high
        self.__w.set_value(np.random.uniform(low=low, high=high, size=[x.shape[1], self.__layer_units]))
        self.__b.set_value(np.zeros(shape=[self.__layer_units]))
        # set parameters queue (weight & bias)
        for i in range(self.max_batch_num):
            self.__w_queue[i].set_value(self.__w.get_value())
            self.__b_queue[i].set_value(self.__b.get_value())

    def do_forward_predict(self, x):
        return np.dot(x, self.__w.get_value()) + self.__b.get_value()

    def do_forward_train(self, x):
        # Performs a forward calculation using the first element of the weight queue
        # and then places it at the end of the queue
        self.__w_queue.append(self.__w_queue.popleft())
        self.__b_queue.append(self.__b_queue.popleft())
        return np.dot(x, self.__w_queue[-1].get_value()) + self.__b_queue[-1].get_value()

    def backward_adjust(self, grad) -> None:
        # Performs a g_w calculation using the first element of the input queue
        g_w = np.dot(self.input_ref.T, grad)
        # Performs a backward adjust using the first element of the weight queue
        # and then places it at the begin of the queue
        self.__w_queue[0].adjust(g_w)
        self.__b_queue[0].adjust(grad)
        self.__w.set_value(self.__w_queue[0].get_value())
        self.__b.set_value(self.__b_queue[0].get_value())

    def backward_propagate(self, grad):
        g_x = np.dot(grad, self.__w_queue[0].get_value().T)
        return g_x

    def __str__(self):
        return "<Dense Layer, Units: {}>".format(self.__layer_units)

    def __repr__(self):
        return "<Dense Layer, Units: {}>".format(self.__layer_units)
