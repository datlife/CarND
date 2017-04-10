"""
--------------------------------------
Week 2: Self-Driving Car Nano-Degree |
--------------------------------------

Author: Dat Nguyen
Homework: Building Mini-flow

Goals:
1. Understand Forward / Back Propagation
2. Understand TensorFlow.
"""


class Neuron:
    """
    *** Abstract Class ***

    Neuron Representation (a Node in Neural Network)
    """
    def __init__(self, inbound_neurons=[]):

        # Neuron from which this Neuron receives values
        self.inbound_neurons = inbound_neurons

        # Neuron which this Neuron sends values:
        self.outbound_neurons = []

        # For each inbound neuron, add this Neuron as an outbound Neuron there
        for n in self.inbound_neurons:
            n.outbound_neurons.append(self)

        # A calculated value
        self.value = None

    def forward(self):
        """
        Forward Propagation

        What this does:
            1. Compute the output value based on inbound_neurons
            2. Store output in self.value
            3. Send output to outbound_neurons

        :return:  send this Neuron's calculated value to Outbound Neurons
        """

        raise NotImplemented

    # def backward(self):
    #     """
    #     Back Propagation
    #     :return:
    #     """
    #     raise NotImplemented


class Input(Neuron):
    """
    Child class of Neuron

    Hold the self.value (data feature, model parameters such as weight /bias)
    No Inbound neurons.
    """

    def __init__(self):
        Neuron.__init__(self)

        # Note: Only Input Neurons have values which could be passed as argument to forward()
        # All other children classes of Neuron should only get the values from inbound_neurons
        # Examples:
        #    val10 = self.inbound_neurons[0].value

    def forward(self, value=None):
        """
        Forward Propagation Implementation of a Neuron
        :param value: input value [default = None]
        :return:
        """
        if value:
            self.value = value


class Add(Neuron):
    """
    Perform calculation on a Neuron

    Input:
    @param x : inbound_neuron
    @param y :

    """
    def __init__(self, x, y):
        Neuron.__init__(self, [x, y])

    def forward(self):
        """
        Calculate input value
        :return:
        """

        for n in self.inbound_neurons:
            if self.value is not None:
                self.value += n.value
            else:
                self.value = n.value


class Linear(Neuron):
    """
    Neuron that performs a Linear Calculation

        OUTPUT = SUM(INPUT*WEIGHT) + BIAS

    """
    def __init__(self, inputs: list, weights: list, bias):

        Neuron.__init__(self, inputs)
        self.weights = weights
        self.bias = bias

    def forward(self):
        """
        Perform linear calculation
        :return: self.value is updated
        """


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    :param feed_dict: dictionary, key = `Input` node and  value = Input
    :return: List of sorted nodes [by values]
    """

    input_neurons = [n for n in feed_dict.keys()]

    graph = {}
    neurons = [n for n in input_neurons]
    while len(neurons) > 0:
        n = neurons.pop(0)
        if n not in graph:
            graph[n] = {'in': set(), 'out': set()}
        for m in n.outbound_neurons:
            if m not in graph:
                graph[m] = {'in': set(), 'out': set()}
            graph[n]['out'].add(m)
            graph[m]['in'].add(n)
            neurons.append(m)

    L = []
    S = set(input_neurons)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_neurons:
            graph[n]['out'].remove(m)
            graph[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(graph[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_neuron, sorted_neurons):
    """
    Performs a forward pass through a list of sorted neurons.

    Arguments:
    ---------
    :param output_neuron: A neuron in the graph, should be the output neuron (have no outgoing edges).
    :param sorted_neurons: a topologically sorted list of neurons.

    :return: the output neuron's value
    """
    for n in sorted_neurons:
        n.forward()
    return output_neuron.value


if __name__ == "__main__":
    # Define two Input Neurons x, y
    x, y = Input(), Input()

    # Define Add Neuron, accepts x and y
    f_add = Add(x, y)

    feed_dict = {x: 10, y: 5}
    sorted_neurons = topological_sort(feed_dict)
    # Return a list of outbound_neurons that add two input neurons
    output = forward_pass(f_add, sorted_neurons)
    print("{} + {} = {} (according to Mini flow)".format(feed_dict[x], feed_dict[y], output))

