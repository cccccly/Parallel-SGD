from typing import Dict

import numpy as np
import os
import hashlib

from nn.dataset.interfaces import AbsDataset


class CIFAR(AbsDataset):

    def __init__(self, check_sum=None):
        self.path = './.data/cifar_data/'
        super().__init__(check_sum)

    def __repr__(self):
        return '<CIFAR-10 classification dataset.>'

    def load(self) -> tuple:
        data: Dict[str, np.ndarray] = np.load(self.path + "cifar10", allow_pickle=True)[()]
        return data["x_train"], data["y_train"].reshape(-1), data["x_test"], data["y_test"].reshape(-1)

    # 处理原数据
    def load_CIFAR_batch(filename):
        with open(filename, 'rb') as f:
            datadict = load_pickle(f)
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            # reshape()是在不改变矩阵的数值的前提下修改矩阵的形状,transpose()对矩阵进行转置
            Y = np.array(Y)
            return X, Y

    # 返回可以直接使用的数据集
    def load_CIFAR10(ROOT):
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))  # os.path.join()将多个路径组合后返回
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)  # 这个函数用于将多个数组进行连接
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    def check_sum(self) -> str:
        if not os.path.exists(self.path + "cifar10"):
            return ''
        sum = hashlib.md5()
        with open(self.path + "cifar10", 'rb') as f:
            sum.update(f.read())
        return sum.hexdigest()

    def extract_files(self) -> list:
        files = [self.path + "cifar10"]
        return files

    def estimate_size(self) -> int:
        return 184380559  #175MB
