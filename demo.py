from typing import List, Iterable, Tuple
import parallel_sgd
import nn
from nn.activation import ReLU, Softmax
from nn.layer import Dense, Conv2D, MaxPool, Flatten, Reshape
import nn.dataset.transforms as transforms

from parallel_sgd.codec.plain import Plain
from nn import IOperator, ITrainable
from nn.layer import Dense, Dropout
from nn.activation import Tanh, Softmax
from nn.model.abstract import Model
from nn.dataset.transforms import ImageCls, Shuffle
from nn.dataset import CIFAR
from nn.dataset import MNIST
from nn.data.numpy_data_feeder import NumpyDataFeeder
import numpy as np


class DNN(Model):
    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        fc1 = Dense(inputs=x, activation=Tanh(), units=784)
        self.__var_list.extend(fc1.variables)

        fc2 = Dense(inputs=fc1, activation=Tanh(), units=784)
        self.__var_list.extend(fc2.variables)

        fc3 = Dense(inputs=fc2, activation=Tanh(), units=392)
        self.__var_list.extend(fc3.variables)

        dropout = Dropout(inputs=fc3)

        fc4 = Dense(inputs=dropout, activation=Tanh(), units=128)
        self.__var_list.extend(fc4.variables)

        fc5 = Dense(inputs=fc4, activation=Softmax(), units=10)
        self.__var_list.extend(fc5.variables)

        return fc5


class DNNPart1(Model):
    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        fc1 = Dense(inputs=x, activation=Tanh(), units=784)
        self.__var_list.extend(fc1.variables)

        fc2 = Dense(inputs=fc1, activation=Tanh(), units=784)
        self.__var_list.extend(fc2.variables)

        fc3 = Dense(inputs=fc2, activation=Tanh(), units=392)
        self.__var_list.extend(fc3.variables)
        return fc3


class DNNPart2(Model):
    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        dropout = Dropout(inputs=x)

        fc4 = Dense(inputs=dropout, activation=Tanh(), units=128)
        self.__var_list.extend(fc4.variables)

        fc5 = Dense(inputs=fc4, activation=Softmax(), units=10)
        self.__var_list.extend(fc5.variables)

        return fc5


if __name__ == '__main__':
    # x = nn.value.Placeholder()
    #
    # l1 = nn.layer.Dense(units=2, inputs=x)
    # l2 = nn.layer.Dense(units=3, inputs=l1)
    #
    # tmp = nn.value.Placeholder()
    # l3 = nn.layer.Dense(units=4, inputs=tmp)
    # l4 = nn.layer.Dense(units=1, inputs=l3)
    # loss = nn.loss.MSELoss()
    #
    # x.set_value(np.asarray([[1, 2], [3, 4]]))
    # mid = l2.F()
    #
    # tmp.set_value(mid)
    # out = l4.F()
    #
    # grad, _ = loss.gradient(out, [[1], [2]])
    # l4.G(grad)
    # mid_grad = tmp.get_gradient()
    #
    # l2.G(mid_grad)

    model1 = DNNPart1()
    model2 = DNNPart2()
    model1.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
    model2.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
    # model.compile(Optimize(nn.optimizer.GDOptimizer, nn.gradient_descent.ADAMOptimizer, gd_params=(1e-3,)))
    model1.compile(nn.gradient_descent.ADAMOptimizer)
    # model2.compile(nn.gradient_descent.ADAMOptimizer)

    # dataset
    trans = Shuffle().add(ImageCls())
    x, y, x_t, y_t = trans(*MNIST().load())
    # x, y, x_t, y_t = MNIST().load()

    x = NumpyDataFeeder(x, y, batch_size=64)
    batch_id = 1

    # train
    for part_x, part_y in x:
        # do forward
        tmp = model1.fit_forward(part_x)
        output = model2.fit_forward(tmp)

        loss = nn.loss.MSELoss()
        grad, _ = loss.gradient(output, part_y)
        print(loss.metric(output, part_y))
        # do backward
        tmp = model2.fit_backward(grad)
        a = tmp.tolist()
        x = np.array(a)
        model1.fit_backward(x)
        batch_id += 1

    # evaluation
    x_t = NumpyDataFeeder(x_t, y_t, batch_size=100)
    acc = 0
    cnt = 0
    for part_x_t, part_y_t in x_t:
        tmp = model1.fit_forward(part_x_t)
        y = model2.fit_forward(tmp)

        y = (y == y.max(axis=1).reshape([-1, 1])).astype('int')
        part_y_t = part_y_t.astype('int')

        result = np.sum(y & part_y_t) / len(y)
        acc += result
        cnt += 1
        # eval_rec = [metric.metric(output, part_y_t) for metric in self.__metrics]
    print("batch{:d}, test_acc:{:.2f}\n".format(batch_id, acc / cnt))
