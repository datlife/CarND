class Input(Layer):
    """
    Input Layer of a Neural Network

    - Input Layer does not have Inbound Layers
    - Input Layer forwards/sends data to
    """
    def __init__(self):
        Layer.__init__(self)

    def forward(self):
        pass