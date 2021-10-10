from typing import List, Tuple, Iterable

import numpy as np

from nn import IOperator, AbsLayer, ITrainable
from nn.model.abstract import Model
from nn.activation.relu import ReLU
from nn.activation.tanh import Tanh
from nn.layer import batchnorm, conv2d, dense, dropout, flatten, maxpool, reshape


class SequentialModel(Model):

    def __init__(self, input_shape=None):
        super().__init__(input_shape=input_shape)
        self.__layers: List[AbsLayer] = []

    def add(self, layer: AbsLayer):
        self.__layers.append(layer)

    def pop(self):
        self.__layers.pop()

    def call(self, x: IOperator) -> IOperator:
        inputs = x
        for layer in self.__layers:
            layer.set_input(inputs)
            inputs = layer
        return inputs

    def trainable_variables(self) -> List[ITrainable]:
        var_list: List[ITrainable] = []
        for layer in self.__layers:
            for item in layer.variables:
                var_list.extend(item)
        return var_list

    def summary(self):
        summary = "\n------------\t\tModel Summary\t\t------------\n"
        for nn in self.__layers:
            nn: AbsLayer
            summary += '\t{}\t\t\n'.format(nn)
            # summary += '\t\tInput:\t{};\n'.format(
            #     [-1] + list(nn.input_ref.shape[1:]) if nn.input_ref is not None else "[Adjust]")
            # summary += '\t\tOutput:\t{};\n'.format(nn.output_shape() if nn.output_shape() else "[Adjust]")
            forward_time = nn.forward_time
            backward_time = nn.backward_time
            val_forward_time = nn.val_forward_time
            summary += '\t\tforward_time_per_batch: {:.4f}\n' \
                       '\t\tbackward_time_per_batch:{:.4f}\n' \
                       '\t\tval_time_per_batch:     {:.4f}\n'\
                .format(np.mean(forward_time), np.mean(backward_time), np.mean(val_forward_time))

        if self.loss:
            summary += '\t------------\t\tAppendix\t\t------------\n'
            summary += '\tLoss:\n\t\t{}\n'.format(self.loss)
            summary += '\tOptimizer:\n\t\t{}\n'.format(self.optimizer)
            summary += '\tMetrics:\n'
            for metric in self.metrics:
                summary += '\t\t<Metric: {}>\n'.format(metric.description())
            summary += '\t------------\t\tAppendix\t\t------------\n'
        summary += '\n------------\t\tModel Summary\t\t------------\n'
        return summary

    def clear(self):
        for layer in self.__layers:
            layer.reset()

    def set_layers_weight(self, weight_list):
        cnt = 0
        for layer in self.__layers:
            if isinstance(layer, conv2d.Conv2D) or isinstance(layer, dense.Dense):
                layer.set_latest_weight(weight_list[cnt])
                cnt += 1

    def get_layers_weight(self) -> list:
        weight_list = []
        for layer in self.__layers:
            if isinstance(layer, conv2d.Conv2D) or isinstance(layer, dense.Dense):
                weight_list.append(layer.get_latest_weight())
        return weight_list

    def set_layers_input(self, input_list: list):
        for layer, item in zip(self.__layers, input_list):
            if isinstance(layer, conv2d.Conv2D) or isinstance(layer, dense.Dense):
                if isinstance(layer.activation, ReLU) or isinstance(layer.activation, Tanh):
                    layer.set_activation_ref_input(item[1])
            elif isinstance(layer, batchnorm.BatchNorm):
                layer.sigma = item[1]
                layer.mu = item[2]
            elif isinstance(layer, dropout.Dropout) or isinstance(layer, maxpool.MaxPool):
                layer.set_mask(item[1])
            layer.set_input_ref(item[0])
        return input_list

    def get_layers_input(self) -> list:
        input_list = []
        for layer in self.__layers:
            if isinstance(layer, conv2d.Conv2D) or isinstance(layer, dense.Dense):
                input_list.append((layer.input_ref[-1],
                                   layer.activation.ref_input[-1] if (isinstance(layer.activation, ReLU)
                                                                      or isinstance(layer.activation, Tanh)) else None))
            elif isinstance(layer, batchnorm.BatchNorm):
                input_list.append((layer.input_ref[-1], layer.sigma, layer.mu))

            elif isinstance(layer, dropout.Dropout) or isinstance(layer, maxpool.MaxPool):
                input_list.append((layer.input_ref[-1], layer.mask[-1]))

            else:
                input_list.append((layer.input_ref[-1], None))
        return input_list

    def clear_layers_input_ref(self):
        for layer in self.__layers:
            layer.clear_input_ref()
            if isinstance(layer, dropout.Dropout) or isinstance(layer, maxpool.MaxPool):
                layer.clear_mask()
            elif isinstance(layer, conv2d.Conv2D) or isinstance(layer, dense.Dense):
                if isinstance(layer.activation, ReLU) or isinstance(layer.activation, Tanh):
                    layer.activation.clear_ref_input()
