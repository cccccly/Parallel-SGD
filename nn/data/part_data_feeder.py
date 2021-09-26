from numpy import ndarray

from nn.data.interface import IDataFeeder


class PartDataFeeder(IDataFeeder):

    def __init__(self, x: ndarray, batch_size: int):
        self.__iter_x: ndarray = x
        self.__batch_size = min(batch_size, len(x))
        self.__batches = len(x) // self.__batch_size
        self.__iter = 0

    @property
    def position(self):
        return self.__iter

    @property
    def length(self):
        return self.__batches

    @property
    def batch_size(self):
        return self.__batch_size

    def __iter__(self):
        for self.__iter in range(self.__batches):
            start = self.__iter * self.__batch_size % (len(self.__iter_x) - self.__batch_size + 1)
            end = start + self.__batch_size
            part_x = self.__iter_x[start:end]
            self.__iter += 1
            yield part_x

    def __repr__(self):
        print(self.__str__())

    def __str__(self):
        return "<Numpy data iterator, current batch: {}, total: {}.>".format(self.__iter, self.__batches)
