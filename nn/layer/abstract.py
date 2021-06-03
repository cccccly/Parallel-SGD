import threading
from abc import abstractmethod
from typing import Iterable, Union

from numpy import ndarray

from nn.activation.interface import IActivation
from nn.activation.linear import Linear
from nn.interface import IOperator, ITrainable, ModelState
from nn.layer.interface import ILazyInitialization


class AbsLayer(IOperator, ILazyInitialization):
    """
        Used for lazy initialization.
    """

    def __init__(self, inputs: IOperator = None, activation: IActivation = None):
        """
            Abstract layer class
        :param inputs: input operator, IOperator instance
        """
        self.__op_input = inputs
        self.__ref_input = None
        self.__activation = activation if activation else Linear()
        self.__initialized = False
        self.__back_propagate_gradient: [ndarray] = None

    @property
    def input_ref(self):
        return self.__ref_input

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
        self.__ref_input = self.__op_input.F(x, state) if self.__op_input else x
        if not self.__initialized:
            self.__forward_prepare(self.__ref_input)
        if state != ModelState.Training:
            return self.__activation.do_forward(self.do_forward_predict(self.__ref_input))
        else:
            return self.__activation.do_forward(self.do_forward_train(self.__ref_input))

    def G(self, grad: [float, ndarray]) -> None:
        """
            Do backward and adjust parameters.
        :param grad: Gradients from back-propagation, set to None when this layer doesnt needs
                input gradients. e.g. loss functions.
        :return: None, try get gradients from placeholder or variable.
        """
        # adjust variables with given gradients.
        gradient = self.__activation.do_backward(None, grad)
        op_running_back_prop = None
        # adjust previous layers.
        if self.__op_input:
            op_running_back_prop = threading.Thread(target=self.__back_propagate, name="BP Thread", args=(grad,))
            op_running_back_prop.start()
        # adjust current layer.
        self.backward_adjust(gradient)
        # join bp
        if op_running_back_prop is not None:
            op_running_back_prop.join()

    def __back_propagate(self, grad):
        """
            Inner back propagate thread
        :return:
        """
        self.__op_input.G(self.backward_propagate(grad))
