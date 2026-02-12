from neuron import Neuron

class Layer:
    def __init__(self, weights_list, biases_list):
        self.neurons = []
        # On suppose que weights_list et biases_list ont la même longueur
        for i in range(len(weights_list)):
            weights = weights_list[i]
            bias = biases_list[i]
            self.neurons.append(Neuron(weights, bias))

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.forward(inputs))
        return outputs