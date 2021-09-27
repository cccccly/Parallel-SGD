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
from nn.data import part_data_feeder, numpy_data_feeder
import numpy as np

import time

import network
import rpc
import nn


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

        return fc2


class DNNPart2(Model):
    def __init__(self, input_shape: [Tuple[int]] = None):
        super().__init__(input_shape)
        self.__var_list: List[ITrainable] = []

    def trainable_variables(self) -> Iterable[ITrainable]:
        return self.__var_list

    def call(self, x: IOperator) -> IOperator:
        self.__var_list: List[ITrainable] = []

        fc3 = Dense(inputs=x, activation=Tanh(), units=392)
        self.__var_list.extend(fc3.variables)

        dropout = Dropout(inputs=fc3)

        fc4 = Dense(inputs=dropout, activation=Tanh(), units=128)
        self.__var_list.extend(fc4.variables)

        fc5 = Dense(inputs=fc4, activation=Softmax(), units=10)
        self.__var_list.extend(fc5.variables)

        return fc5


class ModelPart1(rpc.AbsSimpleExecutor):

    def __init__(self, node_id, worker_group, initializer_id):
        super().__init__(node_id, worker_group, initializer_id)
        self.__train_x = []

    def requests(self) -> list:
        return ["x"]

    def satisfy(self, reply: List[rpc.ReplyPackage]) -> list:
        for data in reply:
            self.__train_x = data.content()
        return []

    def ready(self) -> bool:
        """
            Is the rpc ready for the job.
        """
        # return self.__train_x is not None
        return True

    def run(self, com: rpc.communication.Communication) -> object:
        print('model1 begin run')
        model1 = DNNPart1()
        model1.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
        model1.compile(nn.gradient_descent.ADAMOptimizer)

        trans = ImageCls()
        x, y, x_t, y_t = trans(*MNIST().load())

        x = part_data_feeder.PartDataFeeder(x, 64)
        cnt = 0
        # train process
        for part_x in x:
            cnt += 1
            # do part1 forward and send it immediately
            com.send_one(1, model1.fit_forward(part_x))
            node_id, backward_grad = com.get_one(True, None)

            print("cnt:", cnt, "node_id", node_id, "shape", backward_grad.shape)
            model1.fit_backward(backward_grad)

        # evaluate process
        x_t = part_data_feeder.PartDataFeeder(x_t, 100)
        for part_x_t in x_t:
            com.send_one(1, model1.fit_forward(part_x_t))

        return "model1 successfully finished"


class ModelPart2(rpc.AbsSimpleExecutor):

    def __init__(self, node_id, worker_group, initializer_id):
        super().__init__(node_id, worker_group, initializer_id)
        self.__train_y = []

    def requests(self) -> list:
        return ["y"]

    def satisfy(self, reply: List[rpc.ReplyPackage]) -> list:
        for data in reply:
            self.__train_y = data.content()
        return []

    def ready(self) -> bool:
        """
            Is the rpc ready for the job.
        """
        # return self.__train_y is not None
        return True

    def run(self, com: rpc.communication.Communication) -> object:
        print("model2 begin run")
        model2 = DNNPart2()
        model2.setup(nn.loss.Cross_Entropy_With_Softmax(), nn.metric.CategoricalAccuracy())
        model2.compile(nn.gradient_descent.ADAMOptimizer)

        trans = ImageCls()
        x, y, x_t, y_t = trans(*MNIST().load())

        y = part_data_feeder.PartDataFeeder(y, 64)

        # loss define
        loss = nn.loss.MSELoss()
        cnt = 0
        for part_y in y:

            # get intermediate data of forward pass
            node_id, forward_temp = com.get_one(True, None)
            cnt += 1

            # do part2 forward
            output = model2.fit_forward(forward_temp)
            # calculate gradient
            grad, _ = loss.gradient(output, part_y)
            # do partial backward
            backward_grad = model2.fit_backward(grad)
            print("cnt:", cnt, "node_id", node_id, "loss", loss.metric(output, part_y))
            com.send_one(0, backward_grad)

        y_t = part_data_feeder.PartDataFeeder(y_t, 100)
        # x_t = numpy_data_feeder.NumpyDataFeeder(x_t, y_t, 100)
        acc = 0
        cnt = 0
        for part_y_t in y_t:
            node_id, forward_temp = com.get_one(True, None)
            y_ = model2.fit_forward(forward_temp)

            y_ = (y_ == y_.max(axis=1).reshape([-1, 1])).astype('int')
            part_y_t = part_y_t.astype('int')

            result = np.sum(y_ & part_y_t) / len(y_)
            acc += result
            cnt += 1

        return "model2 successfully finished and test acc {:.3f}".format(acc/cnt)


if __name__ == '__main__':
    # 分配运行节点
    nodes = network.NodeAssignment()
    nodes.add(0, "192.168.31.40")
    nodes.add(1, "192.168.31.152")

    # trans = Shuffle().add(ImageCls())
    # x, y, x_t, y_t = trans(*MNIST().load())
    # x = NumpyDataFeeder(x, y, batch_size=64)
    # x, y, x_t, y_t = MNIST().load()


    def data_set_dispatch(node_id: int, request: object) -> rpc.models.IReplyPackage:
        if request == "x":
            return rpc.ReplyPackage("0")
        elif request == "y":
            return rpc.ReplyPackage("0")


    # 创建执行环境
    with network.Request().request(nodes) as com:
        master = rpc.Coordinator(com)
        master.submit_single(ModelPart1, 0)
        master.submit_single(ModelPart2, 1)
        master.resources_dispatch(data_set_dispatch)
        res, err = master.join()
        # 打印结果
        print(res)
