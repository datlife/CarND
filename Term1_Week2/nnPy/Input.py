from .Layer import Layer


class Input(Layer):
    """
    Input Layer of a Neural Network

    - Input Layer does not have Inbound Layers
    - Input Layer forwards/sends data to
    """
    def __init__(self):
        Layer.__init__(self)

    def forward(self):
        """
        Do nothing because nothing is calculated
        :return:
        """
        pass

    def backward(self):
        # An Input layer has no inputs so the gradient (derivative)
        # is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {self: 0}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbound_layers:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1