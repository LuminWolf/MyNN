import cupy as cp


def softmax(x):
    shiftx = x - cp.max(x)
    exps = cp.exp(shiftx)
    return exps / cp.sum(exps)


def nn_grad_sum(grad1, grad2):
    grad = []
    for layer_num in range(len(grad2)):
        for i in range(len(grad2[layer_num])):
            if len(grad2[layer_num][i]) == 0:
                grad.append([])
            else:
                grad.append(grad1[layer_num][i] + grad2[layer_num][i])
    return grad


def layer_grad_sum(grad1, grad2):
    grad = []
    for i in range(len(grad1)):
        if not grad1:
            grad.append([])
        else:
            grad.append(grad1[i] + grad2[i])
    return grad


def nn_grad_zero(grad):
    for layer_num in range(len(grad)):
        for i in range(len(grad[layer_num])):
            grad[layer_num][i] = cp.zeros_like(grad[layer_num][i])
    return grad
