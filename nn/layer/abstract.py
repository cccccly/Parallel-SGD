import collections
import time
from abc import abstractmethod
from typing import Iterable, Union

from numpy import ndarray

from nn.activation.interface import IActivation
from nn.activation.linear import Linear
from nn.interface import IOperator, ITrainable, ModelState
from nn.layer.interface import ILazyInitialization
import log


class AbsLayer(IOperator, ILazyInitialization):
    """
        Used for lazy initialization.
    """

    def __init__(self, inputs: IOperator = None, activation: IActivation = None, max_batch_num: int = 1):
        """
            Abstract layer class
        :param inputs: input operator, IOperator instance
        self.__max_batch_num : max batch numbers in pipeline and it is generally the same as the number of
        workers/machines
        """
        self.__op_input = inputs
        self.__ref_input = collections.deque()
        self.__activation = activation if activation else Linear()
        self.__initialized = False
        self.__max_batch_num = max_batch_num
        self.__forward_time = []
        self.__backward_time = []
        self.__val_forward_time = []

    @property
    def max_batch_num(self):
        return self.__max_batch_num

    @property
    def forward_time(self):
        return self.__forward_time

    # validation forward_time
    @property
    def val_forward_time(self):
        return self.__val_forward_time

    @property
    def backward_time(self):
        return self.__backward_time

    @property
    def input_ref(self):
        return self.__ref_input[0]

    def set_input(self, inputs: IOperator):
        self.__op_input = inputs

    def __getstate__(self):
        self.__ref_input = None
        return self.__dict__

    @property
    @abstractmethod
    def variables(self) -> Iterable[ITrainable]:
        """
            Trainable units within this scope.
        :return: tuple
        """
        pass

    @abstractmethod
    def initialize_parameters(self, x) -> None:
        """
            Initialize parameters with given input_ref (x)
        :param x: ndarray
        """
        pass

    @abstractmethod
    def do_forward_predict(self, x):
        """
            Do forward propagate with given input_ref.
        :param x: ndarray
        """
        pass

    @abstractmethod
    def do_forward_train(self, x):
        """
            Do forward propagate with given input_ref.
        :param x: ndarray
        """
        pass

    @abstractmethod
    def backward_adjust(self, grad) -> None:
        """
            Backward propagate with weights adjusting.
        :param grad: ndarray
        """
        pass

    @abstractmethod
    def backward_propagate(self, grad):
        """
            Backward propagate.
        :param grad: ndarray
        :return: return the gradient from backward to input_ref (x)
        """
        pass

    def reset(self):
        self.__initialized = False

    def __forward_prepare(self, x):
        self.initialize_parameters(x)
        self.__initialized = True

    def F(self, x: [float, ndarray, tuple] = None, state: ModelState = ModelState.Training) -> Union[float, ndarray]:
        """
            Do forward propagate.
        :param x: input of this layer.
                This parameter works only when this layer is not part of the computation graph.
        :param state: State to identify training process, works in some particular layer like
                (Dropout).
        :return: output of this layer.
        """
        self.__ref_input.append(self.__op_input.F(x, state) if self.__op_input else x)
        if not self.__initialized:
            self.__forward_prepare(self.__ref_input[-1])
        if state != ModelState.Training:
            begin = time.time()
            output = self.__activation.do_forward(self.do_forward_predict(self.__ref_input[-1]), training=False)
            # pop last input because no backward pass in predicting progress
            self.__ref_input.popleft()
            self.__val_forward_time.append(time.time() - begin)
            return output
        else:
            begin = time.time()
            output = self.__activation.do_forward(self.do_forward_train(self.__ref_input[-1]))
            self.__forward_time.append(time.time() - begin)
            return output

    def G(self, grad: [float, ndarray]) -> None:
        """
            Do backward and adjust parameters.
        :param grad: Gradients from back-propagation, set to None when this layer doesnt needs
                input gradients. e.g. loss functions.
        :return: None, try get gradients from placeholder or variable.
        """
        begin = time.time()
        # adjust variables with given gradients.
        gradient = self.__activation.do_backward(None, grad)
        # adjust current layer.
        self.backward_adjust(gradient)
        self.__backward_time.append(time.time() - begin)
        # adjust previous layers.
        if self.__op_input:
            self.__op_input.G(self.backward_propagate(gradient))
        self.__ref_input.popleft()
