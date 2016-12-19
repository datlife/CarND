from .Layer import Layer
import numpy as np


class Sigmoid(Layer):
    """
    Sigmoid Layer for Neural Network

    - Sigmoid Layer has this formula
            sigmoid(x) = 1/(1 + exp(-x)
    - Sigmoid Layer is to smoothly curve the transition from 0 --> 1 in such a way that the function is differentiable
    - Derivative of Sigmoid function is :
            sigmoid'(x) = sigmoid(x)*(1 - sigmoid(x))
    """
    def __init__(self, input_layers):
        Layer.__init__(self, [input_layers])

    def _sigmoid(self, x):
        """
        This method is separate from 'forward' because it later be used as 'backward'
        :param x: numpy array-like object
        :return: result of sigmoid function
        """
        return 1./(1. + np.exp(-x))

    def forward(self):
        """
        Forward Propagation of Sigmoid Layer
        :return: self.outbound_values is updated
        """
        inputs = self.inbound_layers[0].value
        self.value = self._sigmoid(inputs)


