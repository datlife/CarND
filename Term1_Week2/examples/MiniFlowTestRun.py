from Week2.nnPy.miniflow import Input, Add, topological_sort, forward_pass

if __name__ == "__main__":
    # Define two Input Neurons x, y
    x, y = Input(), Input()

    # Define Add Neuron, accepts x and y
    f_add = Add(x, y)

    feed_dict = {x: 10, y: 5}
    sorted_neurons = topological_sort(feed_dict)
    # Return a list of outbound_neurons that add two input neurons
    output = forward_pass(f_add, sorted_neurons)
    print("{} + {} = {} (according to Mini-Flow)".format(feed_dict[x], feed_dict[y], output))
