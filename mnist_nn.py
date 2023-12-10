import numpy as np
import matplotlib.pyplot as plt

import mnist_loader

from mynn import nn
from mynn import data
from mynn import metrics
from mynn import optim


class NeuralNetwork(nn.Module):
    """
    self.layers: 神经网络每层按顺序排列,(不包括损失函数层)
    """
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(784, 10),
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
   
    
def model_test(model, loss):
    test_x = np.array([model.forward(i[0]) for i in test_data])  # 测试
    test_x = np.argmax(test_x, axis=1)
    test_label = np.array([t_data[1] for t_data in test_data])
    confusion_matrix = metrics.get_confusion_matrix(test_x, test_label)  # 绘制混淆矩阵
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    loss.plot_loss()  # 输出训练信息
    loss.plot_acc()
    print(confusion_matrix)
    print(f"训练数据量: {len(training_data)}\n"
          f"验证数据量: {len(validation_data)}\n"
          f"测试数据量: {len(test_label)}\n"
          f"EPOCHS = {EPOCHS}\n"
          f"Accuracy: {accuracy * 100:.3f}%\n")
    

def main():
    loss = train(0)
    loss.plot_loss()
    """
    # criterion = nn.MSELoss()
    # model = NeuralNetwork()
    # criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # optimizer = optim.SGD(model, BATCH_SIZE, LR)  # 随机梯度下降
    # optimizer.lr_scheduler = optim.ExponentialLR(optimizer, 0.99)  # 设置学习率衰减
    # optimizer.weight_decay = L2_LAMBDA  # 设置L2正则化
    # loss = optimizer.train(criterion, training_data, validation_data, EPOCHS)  # 训练
    """
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
    np.set_printoptions(suppress=True)
    EPOCHS = 16
    BATCH_SIZE = 50000
    LR = 2e-5
    L2_LAMBDA = 1e-4
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    print("Dataset loaded")
    main()
