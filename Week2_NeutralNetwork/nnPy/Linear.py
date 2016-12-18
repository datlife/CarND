import numpy as np


class Linear(Layer):
    def __init__(self, inbound_layer, weights, bias):
        Layer.__init__(self, [inbound_layer, weights, bias])

    def forward(self):
        inputs = self.inbound_layers[0].value
        weights = self.inbound_layers[1].value
        bias    = self.inbound_layers[2].value
        self.outbound_layers = np.dot(inputs, weights) + bias
