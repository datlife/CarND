import numpy as np


def sgd_update(trainables, learning_rate=1e-2):
    """
    Stochastic Gradient Descent

    :param trainables: a list of 'Input' layers represent weights/biases
    :param learning_rate: how 'fast' the gradient descent moving toward the slope
    :return:
    """
    for i in trainables:
        grad_cost = i.gradients[i]
        grad_cost *= learning_rate
        i.value -= grad_cost

