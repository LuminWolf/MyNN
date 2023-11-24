import time
import copy
import random

import numpy as np
import matplotlib.pyplot as plt

from . import utils
from . import data
# from . import metrics


class TrainHistory:
    def __init__(self, train_loss, val_loss, train_acc, val_acc):
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.train_acc = train_acc
        self.val_acc = val_acc

    def plot_loss(self):
        plt.plot(range(len(self.train_loss)), self.train_loss, label="train")
        plt.plot(range(len(self.val_loss)), self.val_loss, label="val")
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()

    def plot_acc(self):
        plt.plot(range(len(self.train_acc)), self.train_acc, label="train")
        plt.plot(range(len(self.val_loss)), self.val_acc, label="val")
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        plt.show()


class Optimizer:
    """
    Base class for all optimizers.
    """
    def __init__(self, model, batch_size, lr):
        self.lr = lr
        self.batch_size = batch_size
        self.model = model
        self.params = self.model.params
        self.lr_scheduler = LRScheduler(self)
        self.weight_decay = 0
        self.weight_num = 0
        self.get_weight_num()

    def get_weight_num(self):
        number = 0
        for i in range(len(self.model.params)):
            if len(self.model.params[i]) != 0:
                number += np.size(self.model.params[i][0])
        self.weight_num = number

    def step(self, grads):
        for i in range(len(self.params)):
            for j in range(len(self.params[i])):
                self.params[i][j] = self.params[i][j] - self.lr * grads[i][j]
        return self.params

    def val_loss(self, criterion, val):
        val_loss = 0
        len_val = len(val)
        val_label = []
        for sample in val:
            val_label.append(sample[1])
        val_label = data.onehot_encoder(10, val_label)
        acc_num = 0
        for i in range(len_val):
            predict = self.model.forward(val[i][0])
            loss = criterion.get_loss(predict, val_label[i]) + self.l2_regularization_loss()
            if np.argmax(val_label[i]) == np.argmax(predict):
                acc_num += 1
            val_loss += loss
        val_loss /= len_val
        acc = acc_num / len_val
        return val_loss, acc

    def train(self, criterion, train, val, epochs):
        train_loss_list = []
        return train_loss_list

    def l2_regularization_loss(self) -> float:
        """
        Calculate the L2 regularization term.
        :return: L2 regularization term.
        """
        weight_sum = 0
        for i in range(len(self.model.params)):
            if len(self.model.params[i]) != 0:
                weight_sum += np.sum(np.square(self.model.params[i][0]))
        loss = (self.weight_decay * weight_sum) / (2 * self.weight_num)
        return loss

    def l2_regularization_grad(self):
        """
        Calculate the gradient of the L2 regularization term.
        """
        weight = copy.deepcopy(self.model.params)
        for i in range(len(weight)):
            if len(weight[i]) != 0:
                weight[i][0] *= self.weight_decay / self.weight_num
                weight[i][1] = np.zeros_like(weight[i][1])  # set bias = 0
        return weight


class SGD(Optimizer):
    """
    Implements stochastic gradient descent
    :p
    """
    def __init__(self, model, batch_size, lr):
        super().__init__(model, batch_size, lr)

    def train(self, criterion, train, val, epochs):
        """
        :param criterion: loss function
        :param val: tuple of training data
        :param train: tuple of training data
        :param epochs: number of epoch
        """
        len_train = len(train)
        train_loss_list = []
        val_loss_list = []
        train_acc_list = []
        val_acc_list = []
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = 0
            print(f"Epoch {epoch + 1}/{epochs}", end='')
            # self.lr_scheduler.print_lr()  # 输出学习率
            random.shuffle(train)
            batches = [train[k:k + self.batch_size] for k in range(0, len_train, self.batch_size)]
            acc_num = 0
            for batch in batches:
                for x, label in batch:
                    predict = self.model.forward(x)  # 前向传播
                    loss = criterion.get_loss(predict, label) + self.l2_regularization_loss()
                    # loss = criterion.get_loss(predict, label)
                    if np.argmax(label) == np.argmax(predict):
                        acc_num += 1
                    train_loss += loss
                    self.model.backward(criterion.backward())  # 反向传播
                    # self.model.grads = utils.nn_grad_sum(self.l2_regularization_grad(), self.model.grads)
                self.model.params = self.step(self.model.grads)
                self.model.download_params()
                self.model.grads = utils.nn_grad_zero(self.model.grads)
            train_acc = acc_num / len_train
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss / len_train)
            val_loss, val_acc = self.val_loss(criterion, val)
            val_acc_list.append(val_acc)
            val_loss_list.append(val_loss)
            self.lr_scheduler.step()  # 学习率衰减
            end_time = time.time()
            print(f"|time: {end_time - start_time:.2f}s")
            """
                  f"|train_loss: {train_loss / len_train:.3f}"
                  f"|train_acc: {train_acc * 100:.2f}%"
                  f"|val_loss: {val_loss:.3f}"
                  f"|val_acc: {val_acc * 100:.2f}%"
            """

        train_history = TrainHistory(train_loss_list, val_loss_list, train_acc_list, val_acc_list)
        return train_history


class LRScheduler:
    """
    Adjust the learning rate based on the number of epochs.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._epoch_num = 1

    def step(self):
        self.optimizer.lr = self.optimizer.lr

    def print_lr(self):
        print(f"lr:{self.optimizer.lr:.6f}")


class StepLR(LRScheduler):
    def __init__(self, optimizer, gama=0.99):
        super().__init__(optimizer)
        self._gama = gama

    def step(self):
        self.optimizer.lr *= self._gama
        self._epoch_num += 1


class ExponentialLR(LRScheduler):
    """
    Decays the learning rate by gamma every epoch.
    """
    def __init__(self, optimizer, gama=0.99):
        super().__init__(optimizer)
        self._gama = gama

    def step(self):
        self.optimizer.lr *= self._gama ** self._epoch_num
        self._epoch_num += 1


# class Adam(Optimizer):
#     def __init__(self, model, batch_size, lr):
#         super().__init__(model, batch_size, lr)
#
#     def adam_optimizer(self, v, s):
#         """
#         实现 ADAM 优化算法
#         v：一阶矩矩阵，初始化为 0
#         s：二阶矩矩阵，初始化为 0
#         """
#         beta1 = 0.9  # 一阶矩的指数衰减率
#         beta2 = 0.999  # 二阶矩的指数衰减率
#         epsilon = 1e-8  # 误差项，防止除数为 0
#         # 计算一阶矩和二阶矩的更新值
#         for layer_num in range(len(self.model.grads)):
#             for i in range(len(self.model.grads[layer_num])):
#                 gradients = self.model.grads[layer_num][i]
#                 v_t = beta1 * v + (1 - beta1) * gradients
#                 s_t = beta2 * s + (1 - beta2) * (gradients ** 2)
#                 self.model.grads[layer_num][i] = v_t / (np.sqrt(s_t) + epsilon)
#
#     #     # 初始化一阶矩矩阵和二阶矩矩阵
#     #     v = {'w': np.zeros_like(gradients['w']), 'b': np.zeros_like(gradients['b'])}
#     #     s = {'w': np.zeros_like(gradients['w']), 'b': np.zeros_like(gradients['b'])}
#
#     def train(self, criterion, train, val, epochs):
#         len_train = len(train)
#         train_loss_list = []
#         for epoch in range(epochs):
#             start_time = time.time()
#             train_loss = 0
#             print(f"Epoch {epoch + 1}/{epochs}")
#             random.shuffle(train)
#             batches = [train[k:k + self.batch_size] for k in range(0, len_train, self.batch_size)]
#             for batch in batches:
#                 for x, label in batch:
#                     predict = self.model.forward(x)  # 前向传播
#                     loss = criterion.get_loss(predict, label)
#                     train_loss += loss
#                     self.model.backward(criterion.backward())  # 反向传播
#                 self.model.params = self.step(self.model.grads)
#                 self.model.download_params()
#                 self.model.grads = utils.nn_grad_zero(self.model.grads)
#             val_loss, val_x = self.val_loss(criterion, val)
#             train_loss_list.append(train_loss / len_train)
#             val_label = np.array([v_data[1] for v_data in val])
#             accuracy = np.sum(val_x == val_label) / np.size(val_label)
#             end_time = time.time()
#             print(f"loss: {train_loss / len_train:.6f} / val_loss: {val_loss:.6f}"
#                   f" | acc:{accuracy * 100:.3f}%"
#                   f" | time: {end_time - start_time:.3f}s")
#         return train_loss_list
