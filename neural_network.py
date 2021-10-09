from typing import List, Iterable, Tuple
import nn
from nn import IOperator, ITrainable
from nn.activation import ReLU, Tanh, Softmax
from nn.layer import Dense, Conv2D, MaxPool, Flatten, Dropout, Reshape, BatchNorm
from nn.model.abstract import Model
from nn.model.sequential import SequentialModel


def alexnet():
    model = SequentialModel(input_shape=[-1, 227, 227, 3])
    model.add(Conv2D(strides=[4, 4], padding="VALID", kernel_size=[11, 11], kernel=96, activation=ReLU()))
    model.add(MaxPool(strides=[2, 2], padding="VALID", size=[3, 3]))
    model.add(Conv2D(strides=[1, 1], padding="SAME", kernel_size=[5, 5], kernel=256, activation=ReLU()))
    model.add(MaxPool(strides=[2, 2], padding="VALID", size=[3, 3]))
    model.add(Conv2D(strides=[1, 1], padding="SAME", kernel_size=[3, 3], kernel=384, activation=ReLU()))
    model.add(Conv2D(strides=[1, 1], padding="SAME", kernel_size=[3, 3], kernel=384, activation=ReLU()))
    model.add(Conv2D(strides=[1, 1], padding="SAME", kernel_size=[3, 3], kernel=256, activation=ReLU()))
    model.add(MaxPool(strides=[2, 2], padding="VALID", size=[3, 3]))
    model.add(Flatten())
    model.add(Dense(units=4096, activation=ReLU()))
    model.add(Dense(units=4096, activation=ReLU()))
    model.add(Dense(units=1000, activation=ReLU()))
    model.add(Dense(units=10, activation=Softmax()))
    return model


class AlexNetMnist(Model):
    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        reshape = Reshape(inputs=x, shape=[-1, 28, 28, 1])

        conv1 = Conv2D(inputs=reshape, kernel=96, kernel_size=[11, 11], strides=[1, 1],
                       activation=nn.activation.ReLU(), padding='SAME')
        for item in conv1.variables:
            self.__var_list.extend(item)
        maxpool1 = MaxPool(inputs=conv1, strides=[2, 2], size=(2, 2), padding='SAME')
        bn1 = BatchNorm(inputs=maxpool1)

        conv2 = Conv2D(inputs=bn1, kernel=256, kernel_size=[5, 5], strides=[1, 1],
                       activation=nn.activation.ReLU(), padding='SAME')
        for item in conv2.variables:
            self.__var_list.extend(item)
        maxpool2 = MaxPool(inputs=conv2, strides=[2, 2], size=(2, 2), padding='SAME')
        bn2 = BatchNorm(inputs=maxpool2)

        conv3 = Conv2D(inputs=bn2, kernel=384, kernel_size=[3, 3], strides=[1, 1],
                       activation=nn.activation.ReLU(), padding='SAME')
        for item in conv3.variables:
            self.__var_list.extend(item)
        maxpool3 = MaxPool(inputs=conv3, strides=[2, 2], size=(2, 2), padding='SAME')
        bn3 = BatchNorm(inputs=maxpool3)

        conv4 = Conv2D(inputs=bn3, kernel=384, kernel_size=[3, 3], strides=[1, 1],
                       activation=nn.activation.ReLU(), padding='SAME')
        for item in conv4.variables:
            self.__var_list.extend(item)

        conv5 = Conv2D(inputs=conv4, kernel=256, kernel_size=[3, 3], strides=[1, 1],
                       activation=nn.activation.ReLU(), padding='SAME')
        for item in conv5.variables:
            self.__var_list.extend(item)

        flatten = Flatten(inputs=conv5)

        fc1 = Dense(inputs=flatten, activation=ReLU(), units=4096)
        for item in fc1.variables:
            self.__var_list.extend(item)

        fc2 = Dense(inputs=fc1, activation=ReLU(), units=4096)
        for item in fc2.variables:
            self.__var_list.extend(item)

        fc3 = Dense(inputs=fc2, activation=Softmax(), units=10)
        for item in fc3.variables:
            self.__var_list.extend(item)

        return fc3


class DNN(Model):
    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        fc1 = Dense(inputs=x, activation=Tanh(), units=784)
        for item in fc1.variables:
            self.__var_list.extend(item)

        fc2 = Dense(inputs=fc1, activation=Tanh(), units=784)
        for item in fc2.variables:
            self.__var_list.extend(item)

        fc3 = Dense(inputs=fc2, activation=Tanh(), units=392)
        for item in fc3.variables:
            self.__var_list.extend(item)

        dropout = Dropout(inputs=fc3)

        fc4 = Dense(inputs=dropout, activation=Tanh(), units=128)
        for item in fc4.variables:
            self.__var_list.extend(item)

        fc5 = Dense(inputs=fc4, activation=Softmax(), units=10)
        for item in fc5.variables:
            self.__var_list.extend(item)

        return fc5


def lenet():
    model = SequentialModel()
    model.add(Reshape(shape=[-1, 28, 28, 1]))
    model.add(Conv2D(kernel=6, kernel_size=[5, 5], strides=[1, 1], activation=nn.activation.ReLU()))
    model.add(MaxPool(strides=[2, 2], size=(2, 2), padding='VALID'))
    model.add(BatchNorm())
    model.add(Conv2D(kernel=16, kernel_size=[5, 5], strides=[1, 1], activation=nn.activation.ReLU()))
    model.add(MaxPool(strides=[2, 2], size=(2, 2), padding='VALID'))
    model.add(BatchNorm())
    model.add(Flatten())
    model.add(Dense(activation=ReLU(), units=120))
    model.add(Dense(activation=ReLU(), units=84))
    model.add(Dense(activation=Softmax(), units=10))
    return model


class LeNet(Model):
    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        reshape = Reshape(inputs=x, shape=[-1, 28, 28, 1])

        conv1 = Conv2D(inputs=reshape, kernel=6, kernel_size=[5, 5], strides=[1, 1], activation=nn.activation.ReLU())
        for item in conv1.variables:
            self.__var_list.extend(item)
        maxpool1 = MaxPool(inputs=conv1, strides=[2, 2], size=(2, 2), padding='VALID')

        bn1 = BatchNorm(inputs=maxpool1)

        conv2 = Conv2D(inputs=bn1, kernel=16, kernel_size=[5, 5], strides=[1, 1], activation=nn.activation.ReLU())
        for item in conv2.variables:
            self.__var_list.extend(item)
        maxpool2 = MaxPool(inputs=conv2, strides=[2, 2], size=(2, 2), padding='VALID')

        bn2 = BatchNorm(inputs=maxpool2)

        flatten = Flatten(inputs=bn2)

        fc1 = Dense(inputs=flatten, activation=ReLU(), units=120)
        for item in fc1.variables:
            self.__var_list.extend(item)

        fc2 = Dense(inputs=fc1, activation=ReLU(), units=84)
        for item in fc2.variables:
            self.__var_list.extend(item)

        fc3 = Dense(inputs=fc2, activation=Softmax(), units=10)
        for item in fc3.variables:
            self.__var_list.extend(item)

        return fc3


class Vgg16(Model):
    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        reshape = Reshape(inputs=x, shape=[-1, 32, 32, 3])

        conv1 = Conv2D(inputs=reshape, kernel=64, kernel_size=[3, 3], strides=[1, 1], activation=nn.activation.ReLU())
        for item in conv1.variables:
            self.__var_list.extend(item)
        conv2 = Conv2D(inputs=conv1, kernel=64, kernel_size=[3, 3], strides=[1, 1], activation=nn.activation.ReLU())
        for item in conv2.variables:
            self.__var_list.extend(item)
        maxpool1 = MaxPool(inputs=conv2, strides=[2, 2], size=(2, 2), padding='SAME')

        conv3 = Conv2D(inputs=maxpool1, kernel=128, kernel_size=[3, 3], strides=[1, 1], activation=nn.activation.ReLU())
        for item in conv3.variables:
            self.__var_list.extend(item)
        conv4 = Conv2D(inputs=conv3, kernel=128, kernel_size=[3, 3], strides=[1, 1], activation=nn.activation.ReLU())
        for item in conv4.variables:
            self.__var_list.extend(item)
        maxpool2 = MaxPool(inputs=conv4, strides=[2, 2], size=(2, 2), padding='SAME')

        conv5 = Conv2D(inputs=maxpool2, kernel=256, kernel_size=[3, 3], strides=[1, 1], activation=nn.activation.ReLU())
        for item in conv5.variables:
            self.__var_list.extend(item)
        conv6 = Conv2D(inputs=conv5, kernel=256, kernel_size=[3, 3], strides=[1, 1], activation=nn.activation.ReLU())
        for item in conv6.variables:
            self.__var_list.extend(item)
        maxpool3 = MaxPool(inputs=conv6, strides=[2, 2], size=(2, 2), padding='SAME')

        flatten = Flatten(inputs=maxpool3)
        dropout = Dropout(inputs=flatten)

        fc1 = Dense(inputs=dropout, activation=Softmax(), units=10, max_batch_num=1)
        for item in fc1.variables:
            self.__var_list.extend(item)

        return fc1


# remaining some bug in this model
class FastModel(Model):
    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        reshape = Reshape(inputs=x, shape=[-1, 32, 32, 3])
        # conv1
        conv1 = Conv2D(inputs=reshape, kernel=32, kernel_size=[5, 5], strides=[1, 1], padding='SAME',
                       activation=nn.activation.ReLU())
        for item in conv1.variables:
            self.__var_list.extend(item)
        maxpool1 = MaxPool(inputs=conv1, strides=[2, 2], size=(3, 3), padding='SAME')

        # conv2
        conv2 = Conv2D(inputs=maxpool1, kernel=32, kernel_size=[5, 5], strides=[1, 1], padding='SAME',
                       activation=nn.activation.ReLU())
        for item in conv2.variables:
            self.__var_list.extend(item)
        maxpool2 = MaxPool(inputs=conv2, strides=[2, 2], size=(3, 3), padding='SAME')

        # conv3
        conv3 = Conv2D(inputs=maxpool2, kernel=32, kernel_size=[5, 5], strides=[1, 1], padding='SAME',
                       activation=nn.activation.ReLU())
        for item in conv3.variables:
            self.__var_list.extend(item)
        maxpool3 = MaxPool(inputs=conv3, strides=[2, 2], size=(3, 3), padding='SAME')

        flatten = Flatten(inputs=maxpool3)

        # fc1
        fc1 = Dense(inputs=flatten, activation=Softmax(), units=10)
        for item in fc1.variables:
            self.__var_list.extend(item)

        return fc1
