from .Layer import Layer
import numpy as np


class MSE(Layer):
    """
    The mean squared error cost function.
    Should be used as the last layer for a network
    """
    def __init__(self, y, a):
        Layer.__init__(self, [y, a])

    def forward(self):
        """
        Calculates mean squared error
        :return:
        NOTE: We reshape these to avoid possible matrix/vector broadcast errors.

        For example:
        If we subtract an array of shape (3,) from an array of shape (3,1).
        We get an array of shape(3,3) as the result when we want an array of shape (3,1) instead.

        Making both arrays (3,1) insures the result is (3,1) and does an element-wise subtraction as expected.
        """
        # Flatten the array into vector (n, 1)
        y = self.inbound_layers[0].value.reshape(-1, 1)
        a = self.inbound_layers[1].value.reshape(-1, 1)

        sub_value = y - a
        cost = np.mean(np.square(sub_value))
        self.value = cost

