"""
Given the starting point of any 'x', Gradient Descent should be able to find
the minimum value of x (global minima) using the cost function f(x) defined below
"""


def f(x):
    """
    Quadratic Function
    :param x: random x
    :return:
    """
    return x**2 + 5


def df(x):
    """
    Derivative of 'f' with respect to 'x'
    :param x: random value
    :return:
    """
    return 2*x


def gradient_descent_update(x, gradx, learning_rate):
    """
    Performs a gradient descent update.

        new_x = x - learning_rate*gradient(x)
    :param x:
    :param gradx: gradient-descent at x
    :param learning_rate: the 'force' to push x
    :return:
    """
    # TODO: Implement gradient descent.
    x = x - gradx*learning_rate
    # Return the new value for x
    return x




