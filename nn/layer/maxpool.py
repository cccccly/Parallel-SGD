from typing import Sequence, Tuple

import tensorflow as tf
import numpy as np
import collections
from nn.activation.interface import IActivation
from nn.interface import IOperator
from nn.layer.abstract import AbsLayer


class MaxPool(AbsLayer):

    def __init__(self, strides: Sequence[int], padding: [Sequence[int], str],
                 size: Sequence[int], activation: IActivation = None, inputs: IOperator = None):
        super().__init__(inputs, activation)
        self.__strides: Sequence = strides
        self.__padding: [Sequence, str] = padding
        self.__size: Sequence = size
        self.__mask = collections.deque()
        self.__out_shape = None
        self.__in_shape = None

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

    def popleft_mask(self):
        self.__mask.popleft()

    def initialize_parameters(self, x) -> None:
        self.__in_shape = x.shape

    def do_forward_predict(self, x):
        left = tf.Variable(tf.constant(x, dtype=tf.float32))
        out = tf.nn.max_pool2d(left, self.__size, self.__strides, self.__padding)
        self.__out_shape = out.numpy().shape
        return out.numpy()

    def do_forward_train(self, x):
        left = tf.Variable(tf.constant(x, dtype=tf.float32))
        with tf.GradientTape() as tape:
            out = tf.nn.max_pool2d(left, self.__size, self.__strides, self.__padding)
        self.__mask.append(tape.gradient(out, left))
        self.__out_shape = out.numpy().shape
        return out.numpy()

    def backward_adjust(self, grad) -> None:
        pass

    def backward_propagate(self, grad):
        indices = tf.where(self.__mask.popleft() > 0)
        updates = tf.reshape(tf.constant(grad), (-1))
        shape = tf.constant(self.__in_shape, dtype=tf.int64)
        return tf.scatter_nd(indices, updates, shape).numpy()

    def get_latest_weight(self) -> np.ndarray:
        pass

    def set_latest_weight(self, latest_weight):
        pass

    def weight_avg(self):
        pass

    def output_shape(self) -> Tuple[int]:
        return self.__out_shape

    def __str__(self):
        return "<MaxPool Layer, filter_size: {}>".format(self.__size)

    def __repr__(self):
        return "<MaxPool Layer, filter_size: {}>".format(self.__size)
