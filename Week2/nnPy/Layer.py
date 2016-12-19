class Layer:
    """
    Layer Abstraction of Neural Network

    - A layer can receive inputs from input layers
    - A layer can send/forward outputs to output layers
    - A neural net has : Input Layer ---> Hidden Layer(s) ---> Output layer
    """

    def __init__(self, inbound_layers=[]):
        self.inbound_layers = inbound_layers
        self.value = None
        self.outbound_layers = []
        for layer in inbound_layers:
            layer.outbound_layers.append(self)

    def forward(self):
        raise NotImplementedError

    def backward():
        raise NotImplementedError
