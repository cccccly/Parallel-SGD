from typing import Dict

import numpy as np
import hashlib
import pickle
import platform
import os

from nn.dataset.interfaces import AbsDataset


# 加载序列文件
def load_pickle(f):
    version = platform.python_version_tuple()  # 判断python的版本
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version:{}".format(version))


# 处理原数据
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        # reshape()是在不改变矩阵的数值的前提下修改矩阵的形状,transpose()对矩阵进行转置
        Y = np.array(Y)
        return X, Y


class CIFAR(AbsDataset):

    def __init__(self, check_sum=None):
        self.path = './.data/cifar_data/cifar-10-batches-py/'
        super().__init__(check_sum)

    def __repr__(self):
        return '<CIFAR-10 classification dataset.>'

    def load(self):
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(self.path, 'data_batch_%d' % (b,))  # os.path.join()将多个路径组合后返回
            X, Y = load_cifar_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)  # 这个函数用于将多个数组进行连接
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = load_cifar_batch(os.path.join(self.path, 'test_batch'))
        return Xtr, Ytr, Xte, Yte
    # def load(self) -> tuple:
    #     data: Dict[str, np.ndarray] = np.load(self.path + "cifar10", allow_pickle=True)[()]
    #     return data["x_train"], data["y_train"].reshape(-1), data["x_test"], data["y_test"].reshape(-1)

    def check_sum(self) -> str:
        # if not os.path.exists(self.path + "cifar-10-batches-py"):
        #     return ''
        # sum = hashlib.md5()
        # with open(self.path + "cifar10", 'rb') as f:
        #     sum.update(f.read())
        # return sum.hexdigest()
        return ''

    def extract_files(self) -> list:
        # files = [self.path + "cifar10"]
        # return files
        return []

    def estimate_size(self) -> int:
        return 184380559  # 175MB
