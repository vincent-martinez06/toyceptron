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


# La classe Layer doit :
# Créer et stocker plusieurs objets Neuron
# Appliquer à tous ses neurones un vecteur d’entrée et produire un vecteur de sortie (méthode forward)
# Tous les neurones d’une couche ont le même nombre d’entrées

# Explications :

# weights_list : liste de listes, chaque sous-liste correspond aux poids d’un neurone
# biases_list : liste de biais, un par neurone
# forward(inputs) : applique les entrées à chaque neurone et retourne la liste des résultats

# La boucle for i in range(len(weights_list)) permet d’accéder à chaque paire poids/biais par leur index.
# On crée chaque neurone avec Neuron(weights_list[i], biases_list[i]).

# Partie de main pur tester la couche
# layer = Layer(
#     weights_list=[
#         [0.2, -0.1, 0.4],
#         [-0.4, 0.3, 0.1],
#     ],
#     biases_list=[0.0, 0.1]
# )
# x = [1.0, 2.0, 4.0]
# raw = layer.forward(x)
# print("Couche (valeurs brutes):", raw)