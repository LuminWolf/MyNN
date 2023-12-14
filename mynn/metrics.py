import cupy as cp


def get_confusion_matrix(predict, label):
    confusion_matrix = cp.zeros((10, 10))
    for i in range(len(predict)):
        confusion_matrix[predict[i], label[i]] += 1
    confusion_matrix.astype(cp.int_)
    return confusion_matrix
