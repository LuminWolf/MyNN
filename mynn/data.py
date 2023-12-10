import random

import numpy as np


class Dataset:
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        
    def __len__(self):
        return self.data_tensor.size(0)
    
    def __getitem__(self, index):
        # 返回索引的数据与标签
        return self.data_tensor[index], self.target_tensor[index]


class DataLoaderIter:
    def __init__(self, batches):
        self.batches = batches
        self.count = 0
        self.len = len(batches)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # , label
        if self.count == self.len:
            raise StopIteration
        batch = self.batches[self.count]
        self.count += 1
        return batch

class DataLoader:
    """
    数据集 iterable  向模型输入数据
    """
    def __init__(self, dataset, batch_size, shuffle=False):
        if shuffle:
            random.shuffle(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.batches = []
        self.split(dataset)
    
    def split(self, dataset):
        self.batches = []
        for i in range(0, len(dataset), self.batch_size):
            self.batches.append(dataset[i: i + self.batch_size])
        # batches = [train[k:k + self.batch_size] for k in range(0, len_train, self.batch_size)]
        
    def __iter__(self):
        return DataLoaderIter(self.batches)
    
    def __len__(self):
        return len(self.dataset)


def normalize(x):
    mini = np.min(x)
    return (x - mini) / np.max(x) - mini


def onehot_encoder(num, y_label):
    """
    :param y_label: list of int
    """
    # classification = np.unique(y_label)
    # num = len(classification)
    onehot = []
    for y in y_label:
        y_zero = np.zeros((num,))
        y_zero[y] = 1
        onehot.append(y_zero)
    return np.array(onehot)


# def dataset_split(dataset, train_percent, val_percent, istest):
#     """
#     No shuffle
#     :param dataset:
#     :param train_percent:
#     :param val_percent:
#     :param istest:
#     :return:x_train, y_train, x_val, y_val, x_test, y_test
#     """
#     if istest:
#         test_percent = 1 - train_percent - val_percent
#     x = dataset.data
#     train_index = int(len(x) * train_percent)
#     y = onehot_encoder(dataset.target)
#     x_train = dataset.data[0:train_index]
#     y_train = y[0:train_index]
#     val_index = train_index + int(len(x) * val_percent)
#     x_val = x[train_index:val_index]
#     y_val = y[train_index:val_index]
#     x_test = x[val_index:]
#     y_test = y[val_index:]
#     return x_train, y_train, x_val, y_val, x_test, y_test
