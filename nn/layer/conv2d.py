from typing import Sequence, Optional
import collections
import numpy as np
import tensorflow as tf

from nn.activation.interface import IActivation
from nn.interface import IOperator
from nn.layer.abstract import AbsLayer
from nn.value.trainable import Weights


class Conv2D(AbsLayer):
    """
        Convolve 2D layer.
    """

    def __init__(self,
                 kernel: int,
                 kernel_size: Sequence[int],
                 strides: Optional[Sequence[int]] = None,
                 padding: str = None,
                 activation: IActivation = None,
                 inputs: IOperator = None,
                 max_batch_num: int = 1):
        """
            Currently support "VALID" convolve only.
        :param kernel: Kernel count
        :param kernel_size: kernel size, [height, width]
        :param strides: strides of convolve operation, [height, width]
        :param padding: padding for input x
        :param activation: activation function, None indicates that this layer use linear activation.
        :param inputs: input operator. IOperator instance.
        """
        # Initialize super class
        super().__init__(inputs, activation, max_batch_num)
        # make default strides
        if strides is None:
            self.__strides: Sequence[int] = (1, 1)
        else:
            self.__strides = strides
        # record count, 4th dimension of output
        self.__count_kernel: int = kernel
        # record size
        self.__size_kernel: Sequence[int] = kernel_size
        # make valid padding
        if padding is None or padding == 'VALID':
            self.__padding = 'VALID'
            # back prop padding
            self.__back_prop_padding: Sequence[Sequence[int]] = ((0, 0), (0, 0), (0, 0), (0, 0))
        elif padding == 'SAME':
            self.__padding = 'SAME'
            # valid for BP
            padding_h = (self.__size_kernel[0] * self.__strides[0]) // 2
            padding_w = (self.__size_kernel[1] * self.__strides[1]) // 2
            self.__back_prop_padding = ((0, 0), (padding_h, padding_h), (padding_w, padding_w), (0, 0))
        else:
            raise AssertionError("Padding can only be \'VALID\' or \'SAME\', but {} were given.".format(padding))
        # make weights
        self.__kernel = Weights()
        self.__bias = Weights()
        # weight queue
        self.__kernel_queue = collections.deque([Weights() for i in range(self.max_batch_num)])
        self.__bias_queue = collections.deque([Weights() for i in range(self.max_batch_num)])
        # 4th dimension of input
        self.__count_input: Optional[int] = None
        # output shape
        self.__shape_output: Optional[Sequence[int]] = None
        # get input if possible
        if inputs and inputs.output_shape():
            self.__get_shape_output(inputs.output_shape())

    @property
    def variables(self) -> list:
        res = []
        for w, b in zip(self.__kernel_queue, self.__bias_queue):
            res.append((w, b))
        return res

    def __get_shape_output(self, input_shape: Sequence[int]):
        if self.__padding == 'VALID':
            out_h = 1 + (input_shape[1] - self.__size_kernel[0]) // self.__strides[0]
            out_w = 1 + (input_shape[2] - self.__size_kernel[1]) // self.__strides[1]
        else:
            out_h = input_shape[1]
            out_w = input_shape[2]
        self.__shape_output = (-1, out_h, out_w, self.__count_kernel)
        self.__count_input = input_shape[3]

    def initialize_parameters(self, x: np.ndarray) -> None:
        # update current shape
        self.__get_shape_output(x.shape)
        shape_kernel = (*self.__size_kernel, self.__count_input, self.__shape_output[3])
        nk = self.__size_kernel[0] * self.__size_kernel[1] * self.__count_input
        nk_1 = self.__size_kernel[0] * self.__size_kernel[1] * self.__count_kernel

        high = np.sqrt(6 / (nk + nk_1))
        low = -high
        self.__kernel.set_value(np.random.uniform(low=low, high=high, size=shape_kernel))
        self.__bias.set_value(np.zeros(shape=self.__shape_output[1:]))
        # set parameters queue (kernel & bias)
        for i in range(self.max_batch_num):
            self.__kernel_queue[i].set_value(self.__kernel.get_value())
            self.__bias_queue[i].set_value(self.__bias.get_value())

    def do_forward_predict(self, x: np.ndarray):
        tf_kernel = tf.Variable(tf.constant(self.__kernel.get_value(), dtype=tf.float32))
        # get input
        tf_input = tf.Variable(tf.constant(x, dtype=tf.float32))
        # get output, 60% of time consumed
        tf_out = tf.nn.conv2d(tf_input, tf_kernel, self.__strides, self.__padding)
        out = tf_out.numpy().astype('float64')
        return out + self.__bias.get_value()

    def do_forward_train(self, x):
        tf_kernel = tf.Variable(tf.constant(self.__kernel_queue[-1].get_value(), dtype=tf.float32))
        # get input
        tf_input = tf.Variable(tf.constant(x, dtype=tf.float32))
        # get output, 60% of time consumed
        tf_out = tf.nn.conv2d(tf_input, tf_kernel, self.__strides, self.__padding)
        out = tf_out.numpy().astype('float64') + self.__bias_queue[-1].get_value()
        return out

    def backward_adjust(self, grad) -> None:
        # rearrange input
        tf_input = tf.transpose(tf.constant(self.input_ref[0], dtype=tf.float32), perm=[3, 1, 2, 0])  # magic number
        # rearrange gradient, 38% of time consumed here
        tf_grad = tf.transpose(tf.constant(grad, dtype=tf.float32), perm=[1, 2, 0, 3])  # magic number, dont change
        # get output
        tf_conv = tf.nn.conv2d(tf_input, tf_grad, self.__strides, self.__back_prop_padding)
        tf_out = tf.transpose(tf_conv, perm=[1, 2, 0, 3])
        out = tf_out.numpy()
        self.__kernel_queue.append(self.__kernel_queue.popleft())
        self.__kernel_queue[-1].adjust(out)
        self.__bias_queue.append(self.__bias_queue.popleft())
        self.__bias_queue[-1].adjust(grad)
        self.__kernel.set_value(self.__kernel_queue[-1].get_value())
        self.__bias.set_value(self.__bias_queue[-1].get_value())

    def backward_propagate(self, grad):
        tf_kernel = tf.constant(self.__kernel_queue[0].get_value(), dtype=tf.float32)
        tf_grad = tf.constant(grad, dtype=tf.float32)
        tf_out = tf.nn.conv2d_transpose(tf_grad, tf_kernel, self.input_ref[0].shape, self.__strides, self.__padding)
        grad = tf_out.numpy()
        return grad.astype('float64')

    def set_latest_weight(self, weight: list):
        for i in range(self.max_batch_num):
            self.__kernel_queue[i].set_value(weight[i][0])
            self.__bias_queue[i].set_value(weight[i][1])

    def get_weight_queue(self) -> list:
        weight = []
        for i in range(self.max_batch_num):
            weight.append((self.__kernel_queue[i].get_value(), self.__bias_queue[i].get_value()))
        return weight


    def weight_avg(self):
        kernel = []
        bias = []
        for i in range(self.max_batch_num):
            kernel.append(self.__kernel_queue[i].get_value())
            bias.append(self.__bias_queue[i].get_value())
        kernel_avg = np.mean(kernel, axis=0)
        bias_avg = np.mean(bias, axis=0)
        for i in range(self.max_batch_num):
            self.__kernel_queue[i].set_value(kernel_avg)
            self.__bias_queue[i].set_value(bias_avg)

    def output_shape(self) -> [list, tuple, None]:
        return self.__shape_output

    def __str__(self):
        return "<2D Convolution Layer, kernel size: {}, count: {}>".format(self.__size_kernel, self.__count_kernel)

    def __repr__(self):
        return "<2D Convolution Layer, kernel size: {}, count: {}>".format(self.__size_kernel, self.__count_kernel)
