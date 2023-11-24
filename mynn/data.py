import numpy as np


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
