"""
MNIST手写数字识别
"""

import numpy as np

import mnist_loader

import mynn.nn as nn
import mynn.metrics as metrics
import mynn.optim as optim


class NeuralNetwork(nn.Module):
    """
    self.layers: 神经网络每层按顺序排列,(不包括损失函数层)
    """
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(784, 10),
                       nn.CrossEntropyLoss(False)]
        self.init()


def main():
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    model = NeuralNetwork()
    optimizer = optim.SGD(model, BATCH_SIZE, LR)
    optimizer.lr_scheduler = optim.ExponentialLR(optimizer, 0.95)  # 设置学习率衰减
    optimizer.weight_decay = L2_LAMBDA  # 设置L2正则化
    loss = optimizer.train(criterion, training_data, validation_data, EPOCHS)  # 训练
    # 测试
    test_x = np.array([model.forward(i[0]) for i in test_data])
    test_x = np.argmax(test_x, axis=1)
    test_label = np.array([t_data[1] for t_data in test_data])
    # 绘制混淆矩阵
    confusion_matrix = metrics.get_confusion_matrix(test_x, test_label)
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    # 输出训练信息
    loss.plot_loss()
    print(confusion_matrix)
    print(f"训练数据量: {len(training_data)}\n"
          f"验证数据量: {len(validation_data)}\n"
          f"测试数据量: {len(test_label)}\n"
          f"EPOCHS = {EPOCHS}\n"
          f"LR = {LR}\n"
          f"Accuracy: {accuracy * 100:.6f}%\n")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    EPOCHS = 10
    BATCH_SIZE = 100000
    LR = 1e-1
    L2_LAMBDA = 1e-4
    # L2_LAMBDA = 0
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("Dataset loaded")
    main()


