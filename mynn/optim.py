import time
import random

import numpy as np
import matplotlib.pyplot as plt

from . import utils
# from . import metrics


class TrainHistory:
    def __init__(self, loss, val_loss):
        self.loss_list = loss
        self.val_loss = val_loss

    def plot_loss(self):
        fig, axes = plt.subplots(2, 1, sharex="col")
        axe1 = axes[0]
        axe2 = axes[1]
        axe1.plot(range(len(self.loss_list)), self.loss_list, label="train")  # 绘制损失
        axe1.set_xlabel("Epochs")
        axe1.set_ylabel("Loss")
        axe1.legend()
        axe2.plot(range(len(self.val_loss)), self.val_loss, label="val")
        axe2.set_xlabel("Epochs")
        axe2.set_ylabel("Loss")
        axe2.legend()
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

    def step(self, grads):
        for i in range(len(self.params)):
            for j in range(len(self.params[i])):
                self.params[i][j] = self.params[i][j] - self.lr * grads[i][j]
        return self.params

    def val_loss(self, model, criterion, val):
        # val_loss
        val_loss = 0
        len_val = len(val)
        val_x = []
        for i in range(len_val):
            predict = model.forward(val[i][0])
            val_x.append(np.argmax(predict))
            loss = criterion.get_loss(predict, val[i][1])
            val_loss += loss
        val_loss /= len_val
        return val_loss, val_x

    def train(self, criterion, train, val, epochs):
        train_loss_list = []
        return train_loss_list

    def l2_regularization(self):
        weight_sum = 0
        number = 0
        for i in range(len(self.model.params)):
            if len(self.model.params[i]) != 0:
                weight_sum += np.sum(np.square(self.model.params[i][0]))
                number += np.size(self.model.params[i][0])
        return (self.weight_decay * weight_sum) / (2 * number)


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
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = 0
            print(f"Epoch {epoch + 1}/{epochs}", end=' ')
            # self.lr_scheduler.print_lr()  # 输出学习率
            random.shuffle(train)
            batches = [train[k:k + self.batch_size] for k in range(0, len_train, self.batch_size)]
            for batch in batches:
                for x, label in batch:
                    predict = self.model.forward(x)  # 前向传播
                    loss = criterion.get_loss(predict, label) + self.l2_regularization()
                    train_loss += loss
                    self.model.backward(criterion.backward())  # 反向传播
                self.model.params = self.step(self.model.grads)
                self.model.download_params()
                self.model.grads = utils.nn_grad_zero(self.model.grads)
            print(self.l2_regularization())
            train_loss_list.append(train_loss / len_train)
            val_loss, val_x = self.val_loss(self.model, criterion, val)
            val_loss_list.append(val_loss)
            val_label = np.array([v_data[1] for v_data in val])
            accuracy = np.sum(val_x == val_label) / np.size(val_label)
            self.lr_scheduler.step()  # 学习率衰减
            end_time = time.time()
            print(f"train_loss: {train_loss / len_train:.6f} | val_loss: {val_loss:.6f}"
                  f" | acc:{accuracy * 100:.3f}%"
                  f" | time: {end_time - start_time:.3f}s")
        train_history = TrainHistory(train_loss_list, val_loss_list)
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


class Adam(Optimizer):
    def __init__(self, model, batch_size, lr):
        super().__init__(model, batch_size, lr)

    def adam_optimizer(self, v, s):
        """
        实现 ADAM 优化算法
        v：一阶矩矩阵，初始化为 0
        s：二阶矩矩阵，初始化为 0
        """
        beta1 = 0.9  # 一阶矩的指数衰减率
        beta2 = 0.999  # 二阶矩的指数衰减率
        epsilon = 1e-8  # 误差项，防止除数为 0
        # 计算一阶矩和二阶矩的更新值
        for layer_num in range(len(self.model.grads)):
            for i in range(len(self.model.grads[layer_num])):
                gradients = self.model.grads[layer_num][i]
                v_t = beta1 * v + (1 - beta1) * gradients
                s_t = beta2 * s + (1 - beta2) * (gradients ** 2)
                self.model.grads[layer_num][i] = v_t / (np.sqrt(s_t) + epsilon)

    #     # 初始化一阶矩矩阵和二阶矩矩阵
    #     v = {'w': np.zeros_like(gradients['w']), 'b': np.zeros_like(gradients['b'])}
    #     s = {'w': np.zeros_like(gradients['w']), 'b': np.zeros_like(gradients['b'])}

    def train(self, criterion, train, val, epochs):
        len_train = len(train)
        train_loss_list = []
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = 0
            print(f"Epoch {epoch + 1}/{epochs}")
            random.shuffle(train)
            batches = [train[k:k + self.batch_size] for k in range(0, len_train, self.batch_size)]
            for batch in batches:
                for x, label in batch:
                    predict = self.model.forward(x)  # 前向传播
                    loss = criterion.get_loss(predict, label)
                    train_loss += loss
                    self.model.backward(criterion.backward())  # 反向传播
                self.model.params = self.step(self.model.grads)
                self.model.download_params()
                self.model.grads = utils.nn_grad_zero(self.model.grads)
            val_loss, val_x = self.val_loss(criterion, val)
            train_loss_list.append(train_loss / len_train)
            val_label = np.array([v_data[1] for v_data in val])
            accuracy = np.sum(val_x == val_label) / np.size(val_label)
            end_time = time.time()
            print(f"loss: {train_loss / len_train:.6f} / val_loss: {val_loss:.6f}"
                  f" | acc:{accuracy * 100:.3f}%"
                  f" | time: {end_time - start_time:.3f}s")
        return train_loss_list
