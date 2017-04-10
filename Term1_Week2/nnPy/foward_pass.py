def forward_pass(output_layer, sorted_layers):
    """
    Performs a forward pass through a list of sorted Layers.

    Arguments:

        `output_layer`: A Layer in the graph, should be the output layer (have no outgoing edges).
        `sorted_layers`: a topologically sorted list of layers.

    Returns the output layer's value
    """

    for n in sorted_layers:
        n.forward()

    return output_layer.value
