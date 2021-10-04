from typing import List, Iterable, Tuple
import nn
from nn import IOperator, ITrainable
from nn.activation import ReLU, Tanh, Softmax
from nn.layer import Dense, Conv2D, MaxPool, Flatten, Dropout, Reshape, BatchNorm
from nn.model.abstract import Model


class DNNPart1(Model):
    def __init__(self, input_shape: [Tuple[int]] = None, max_batch_num=1):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []
        self.max_batch_num = max_batch_num

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        fc1 = Dense(inputs=x, activation=Tanh(), units=784, max_batch_num=self.max_batch_num)
        for item in fc1.variables:
            self.__var_list.extend(item)

        fc2 = Dense(inputs=fc1, activation=Tanh(), units=784, max_batch_num=self.max_batch_num)
        for item in fc2.variables:
            self.__var_list.extend(item)

        return fc2


class DNNPart2(Model):
    def __init__(self, input_shape: [Tuple[int]] = None, max_batch_num=1):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []
        self.max_batch_num = max_batch_num

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        fc3 = Dense(inputs=x, activation=Tanh(), units=392, max_batch_num=self.max_batch_num)
        for item in fc3.variables:
            self.__var_list.extend(item)

        dropout = Dropout(inputs=fc3)

        fc4 = Dense(inputs=dropout, activation=Tanh(), units=128, max_batch_num=self.max_batch_num)
        for item in fc4.variables:
            self.__var_list.extend(item)

        fc5 = Dense(inputs=fc4, activation=Softmax(), units=10)
        for item in fc5.variables:
            self.__var_list.extend(item)

        return fc5


class LeNetPart1(Model):
    def __init__(self, input_shape: [Tuple[int]] = None, max_batch_num=1):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []
        self.max_batch_num = max_batch_num

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        reshape = Reshape(inputs=x, shape=[-1, 28, 28, 1])

        conv1 = Conv2D(inputs=reshape, kernel=6, kernel_size=[5, 5], strides=[1, 1]
                       , activation=nn.activation.ReLU(), max_batch_num=self.max_batch_num)
        for item in conv1.variables:
            self.__var_list.extend(item)
        maxpool1 = MaxPool(inputs=conv1, strides=[2, 2], size=(2, 2), padding='VALID')

        bn1 = BatchNorm(inputs=maxpool1)

        conv2 = Conv2D(inputs=bn1, kernel=16, kernel_size=[5, 5], strides=[1, 1],
                       activation=nn.activation.ReLU(), max_batch_num=self.max_batch_num)
        for item in conv2.variables:
            self.__var_list.extend(item)
        maxpool2 = MaxPool(inputs=conv2, strides=[2, 2], size=(2, 2), padding='VALID')

        bn2 = BatchNorm(inputs=maxpool2)

        return bn2


class LeNetPart2(Model):
    def __init__(self, input_shape: [Tuple[int]] = None, max_batch_num=1):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []
        self.max_batch_num = max_batch_num

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        flatten = Flatten(inputs=x)

        fc1 = Dense(inputs=flatten, activation=ReLU(), units=120, max_batch_num=self.max_batch_num)
        for item in fc1.variables:
            self.__var_list.extend(item)

        fc2 = Dense(inputs=fc1, activation=ReLU(), units=84, max_batch_num=self.max_batch_num)
        for item in fc2.variables:
            self.__var_list.extend(item)

        fc3 = Dense(inputs=fc2, activation=Softmax(), units=10, max_batch_num=self.max_batch_num)
        for item in fc3.variables:
            self.__var_list.extend(item)

        return fc3