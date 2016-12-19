from .Layer import Layer


class Input(Layer):
    """
    Input Layer of a Neural Network

    - Input Layer does not have Inbound Layers
    - Input Layer forwards/sends data to
    """
    def __init__(self):
        Layer.__init__(self)
        self.gradient = {self: 0}

    def forward(self):
        """
        Do nothing because nothing is calculated
        :return:
        """
        pass

    def backward(self):
        """
        An Input Layer has no inputs so we refer to itself for the gradient
        :return:
        """
        for n in self.outbound_layers:
            self.gradient[self] += n.gradient[self]
