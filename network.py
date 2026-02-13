from layer import Layer

class Network:
    def __init__(self, input_size, activation):
        self.input_size = input_size
        self.activation = activation
        self.layers = []

    def add(self, weights, biases):
        # Ajoute une couche au réseau
        layer = Layer(weights, biases)
        self.layers.append(layer)

    def feedforward(self, inputs):
        # Fait circuler les inputs à travers toutes les couches
        current = inputs
        for layer in self.layers:
            # On applique la couche, puis la fonction d'activation à chaque sortie
            raw_outputs = layer.forward(current)
            current = [self.activation(o) for o in raw_outputs]
        return current

# Explications

# input_size : nombre d’entrées du réseau (utile pour vérifier la cohérence)
# activation : fonction d’activation (ex : sigmoïde)
# layers : liste des couches du réseau
# add(weights, biases) : ajoute une couche (les poids et biais sont des listes)
# feedforward(inputs) : applique chaque couche successivement, en activant les sorties à chaque étape

# Conclusion

# Partie de main pour tester le réseau:
# net = Network(input_size=3, activation=act_sigmoid)
# net.add(weights=[...], biases=[...])
# ...
# y = net.feedforward(x)
# print("Sorties activées :", y)

# Vous pouvez maintenant construire et tester un réseau de neurones simple, couche par couche, avec votre Toyceptron.