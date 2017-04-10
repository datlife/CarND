import numpy as np


def soft_max(arr):
    """
    Calculate probability of each elements in array

    :param arr: array-like
    :return: array : same size as arr with probabilities
    """
    total = np.sum(np.exp(arr), axis=0)
    softmax = np.exp(arr)/total
    return softmax

# Test run
logits = [[3.0, 1.0, 0.2], [2.1, 1.2, 3.0]]
print(soft_max(logits))
