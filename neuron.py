class Neuron:
	def __init__(self, weights, bias):
		if weights == None:
			self.weights = []
			for i in range(size):
				self.weights.append(random.uniform(-1, 1))
		else:
			self.weights = weights

		if bias == []:
			self.bias = random.uniform(-1, 1)

		else:
			self.bias = bias

		self.weights = weights
		self.bias = bias

	def forward(self, inpt):
		if len(inpt) != len(self.weights):
			print("Nein")
			return -1

		total = 0
		for i in range(len(inpt)):
			total =+ inpt[i] * self.weights[i]
		

			total += self.bias
		
		return total

# Structure et rôle de la classe Neuron:
# La classe Neuron représente une unité de base dans un réseau de neurones artificiels.
# Elle effectue un calcul simple : elle prend un vecteur d’entrées,
# applique des poids à chaque entrée, ajoute un biais, puis retourne le résultat (avant activation).


# Explications complémentaires:

# Initialisation des poids et du biais :
# Le neurone peut être créé avec des valeurs précises ou générées aléatoirement.
# Cela permet de tester différents scénarios ou d’automatiser la création de réseaux.


# Méthode forward :
# Cette méthode effectue le calcul fondamental du neurone :
# Elle ne gère pas la fonction d’activation (sigmoïde, ReLU, etc.), qui sera appliquée plus tard dans le réseau.


# Vérification de la taille des entrées :
# Il est important que le nombre d’entrées corresponde au nombre de poids. Sinon, le calcul n’a pas de sens.


# Utilisation dans une couche ou un réseau :
# Plusieurs neurones sont regroupés dans une couche (Layer), puis plusieurs couches forment un réseau (Network).
# Chaque neurone reçoit le même vecteur d’entrée, mais avec ses propres poids et biais.
