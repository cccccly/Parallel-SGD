from typing import List, Tuple, Iterable

import numpy as np

from nn import IOperator, AbsLayer, ITrainable
from nn.model.abstract import Model


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
