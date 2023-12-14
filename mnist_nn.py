# import numpy as np

import mnist_loader

from mynn import nn
from mynn import data
# from mynn import metrics
from mynn import optim


class NeuralNetwork(nn.Module):
    """
    self.layers: 神经网络每层按顺序排列,(不包括损失函数层)
    """
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(784, 20),
                       nn.Sigmoid(),
                       nn.Linear(20, 10),
                       nn.CrossEntropyLoss(False)]
        self.init()


def train(parameter) -> optim.TrainHistory:
    model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model, BATCH_SIZE, LR)
    optimizer.weight_decay = parameter
    train = data.DataLoader(training_data, BATCH_SIZE)
    val = data.DataLoader(validation_data, BATCH_SIZE)
    loss = optimizer.train(criterion, train, validation_data, EPOCHS)
    return loss


def main():
    # criterion = nn.MSELoss()
    model = NeuralNetwork()
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(model, BATCH_SIZE, LR)  # 随机梯度下降
    optimizer.lr_scheduler = optim.ExponentialLR(optimizer, 0.99)  # 设置学习率衰减
    optimizer.weight_decay = L2_LAMBDA  # 设置L2正则化
    train_history = optimizer.train(criterion, training_data, validation_data, EPOCHS)  # 训练
    train_history.plot_loss()

    """
    parameter_list = [1e-2, 1e-3, 1e-4]
    loss_list = [] # 1e-1, 1e-2, 1e-3, 1e-4
    for i in range(len(parameter_list)):
        loss_list.append(train(parameter_list[i]))
    # 绘制训练信息
    fig, ax = plt.subplots()
    ax.set_xlabel("Eppochs")
    ax.set_ylabel("Loss")
    ax.plot(range(len(loss_list[0])), loss_list[0], label="lambd=1")
    ax.plot(range(len(loss_list[1])), loss_list[1], label="lambda=1e-1")
    ax.plot(range(len(loss_list[2])), loss_list[2], label="lambda=1e-2")
    ax.legend()
    plt.show()
    """


if __name__ == "__main__":
    # np.set_printoptions(suppress=True)
    EPOCHS = 16
    BATCH_SIZE = 50000
    LR = 2e-5
    L2_LAMBDA = 1e-5
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("Dataset loaded")
    main()
