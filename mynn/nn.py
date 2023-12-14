"""
1.BN层
2.Residual Block
3.Kaiming初始化, Xavier初始化
"""
import cupy
import cupy as cp

from . import utils


class Layer:
    def __init__(self):
        self.params = []
        self.grads = []


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        out = cp.maximum(0, x)
        self.out = out
        return out

    def backward(self, dout):
        grad = self.out.copy()
        grad[grad > 0] = 1.0
        grad[grad < 0] = 0.0
        dx = dout * grad
        return dx


class LeakyReLU(Layer):
    def __init__(self, negative_slope=0.18):
        super().__init__()
        self.out = None
        self.negative_slope = negative_slope

    def forward(self, x):
        out = cp.maximum(self.negative_slope * x, x)
        self.out = out
        return out

    def backward(self, dout):
        grad = self.out.copy()
        grad[grad > 0] = 1
        grad[grad < 0] = self.negative_slope
        dx = dout * grad
        return dx


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x):
        out = 1 / (1 + cp.exp(-x))
        self.out = out
        return out

    def backward(self, grad):
        sigmoid_grad = self.out * (1 - self.out)
        return grad * sigmoid_grad


class Linear(Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.init_params(in_features, out_features)
        self.grads = [cp.zeros_like(self.params[0]), cp.zeros_like(self.params[1])]
        self.x = None

    @staticmethod
    def kaiming_uniform(in_features, out_features):
        weight = cp.random.normal(0, 2 / out_features, (in_features, out_features))
        return weight

    def init_params(self, in_features, out_features):
        weight = self.kaiming_uniform(in_features, out_features)
        bias = cp.random.normal(size=(out_features,))
        self.params = [weight, bias]

    def forward(self, x):
        w, b = self.params
        out = cp.dot(x, w) + b
        self.x = x
        return out

    def backward(self, grad):
        w, b = self.params
        dx = cp.dot(grad, w.T)
        dw = cp.outer(self.x.T, grad)
        db = cp.sum(grad, axis=0)
        # 梯度累加
        grad = utils.layer_grad_sum(self.grads, [dw, db])
        self.grads[0][...] = grad[0]
        self.grads[1][...] = grad[1]
        return dx


class Loss(Layer):
    def __init__(self):
        """
        :param self.predicted: 预测值
        :param self.label: 标签
        """
        super().__init__()
        self.predict = None
        self.label = None

    def get_loss(self, predict, label):
        self.predict = predict
        self.label = label
        return None

    def backward(self, dout=None):
        return None


class MSELoss(Loss):
    """
    get_loss:
    """
    def __init__(self):
        super().__init__()
        self.predict = None
        self.label = None

    def get_loss(self, predict, label):
        self.predict = predict
        self.label = label
        out = (cp.sum(cp.square(label - predict)) / cp.size(label))
        return out

    def backward(self, dout=None):
        n = self.label.shape[0]
        grad = (2 / n) * (self.predict - self.label)
        return grad


class CrossEntropyLoss(Loss):
    """
    CrossEntropyLoss with softmax as activation function
    """
    def __init__(self, isloss=True):
        """
        :param isloss: 是否为损失函数,False代表其为softmax激活函数
        """
        super().__init__()
        self.isloss = isloss

    def get_loss(self, predict, label):
        """
        :param predict: 预测值
        :param label: 标签值
        :return: 损失值
        """
        # self.predicted = utils.softmax(predict)
        self.predict = cp.clip(predict, a_min=1e-8, a_max=None)
        self.label = label
        out = -cp.sum(cp.log(self.predict) * label) / len(self.predict)
        return out

    def forward(self, x):
        self.predict = utils.softmax(x)
        return self.predict

    def backward(self, dout=None):
        if self.isloss:
            grad = self.predict - self.label
        else:
            grad = dout
        return grad


class Module:
    def __init__(self):
        self.layers = []
        self.params = []
        self.grads = []

    def init(self):
        self.upload_params()
        self.upload_grads()

    def upload_params(self):
        for layer in self.layers:
            self.params.append(layer.params)

    def download_params(self):
        for i in range(len(self.layers)):
            self.layers[i].params = self.params[i]

    def upload_grads(self):
        self.grads = []
        for layer in self.layers:
            self.grads.append(layer.grads)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i].forward(x)
        return x

    def backward(self, dout):
        # 反向传播
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        self.upload_grads()
