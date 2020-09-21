from abc import ABCMeta, abstractmethod

from numpy import ndarray

from nn.data.interface import IDataFeeder
from nn.interface import IOptimizer
from nn.loss.abstract import ILoss
from nn.metric import IMetric
from utils.log import IPrinter
from nn.model.utils import FitResultHelper


class IModel(metaclass=ABCMeta):

    @abstractmethod
    def compile(self, optimizer:IOptimizer, loss:ILoss, *metrics:IMetric):
        """
            Compile model with given optimizer, loss and metrics
        :param optimizer: IOptimizer
        :param loss: ILoss
        :param metrics: IMetric
        :return: None
        """
        pass

    @abstractmethod
    def fit(self, x:[ndarray, IDataFeeder], epoch:int, label:[ndarray]=None, batch_size:int=64, printer:IPrinter=None) -> FitResultHelper:
        """
            Fit model with given samples.
        :param x: ndarray or data feeder. requires a IDataFeeder instance or both x and label for ndarray instance.
        :param epoch: int, Epoch of training
        :param label: ndarray, Label of samples
        :param batch_size: int, batch size
        :param printer: printer type
        :return: Fitting result, contains all history records.
        """
        pass

    @abstractmethod
    def evaluate(self, x:ndarray, label:ndarray):
        """
            Evaluate this model with given metric.
        :param x: input samples
        :param label: labels
        :return: evaluation result
        """
        pass

    @abstractmethod
    def predict(self, x:ndarray):
        """
            Predict give input
        :param x: input samples
        :return:
        """
        pass

    @abstractmethod
    def clear(self):
        """
            Clear and reset model parameters.
        :return:
        """
        pass