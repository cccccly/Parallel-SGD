import numpy as np
import collections
from nn.activation.abstract import IActivation
from nn.layer.abstract import AbsLayer


class Dropout(AbsLayer):

    def __init__(self, drop_out_rate: float = 0.5, activation: IActivation = None, inputs=None):
        super().__init__(inputs, activation)
        self.__mask = collections.deque()
        self.__probability = drop_out_rate
        self.__scale = 1 / (1 - drop_out_rate)

    @property
    def variables(self) -> tuple:
        return ()

    @property
    def mask(self):
        return self.__mask

    def set_mask(self, value):
        self.__mask.append(value)

    def clear_mask(self):
        self.__mask.clear()

    def initialize_parameters(self, x) -> None:
        pass

    def do_forward_predict(self, x):
        return x

    def do_forward_train(self, x):
        self.__mask.append(np.random.uniform(0, 1, size=x.shape) > self.__probability)
        return np.multiply(x, self.__mask[-1]) * self.__scale

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        return np.multiply(grad, self.__mask.popleft()) * self.__scale

    def get_latest_weight(self) -> np.ndarray:
        pass

    def set_latest_weight(self, latest_weight):
        pass

    def weight_avg(self):
        pass

    def output_shape(self) -> [list, tuple, None]:
        return None

    def __str__(self):
        return "<Dropout Layer, Drop probability: {}>".format(self.__probability)

    def __repr__(self):
        print(self.__str__())
