from math import exp

def act_identity(x):
    """Fonction identité : retourne la valeur telle quelle."""
    return x

def act_threshold(x):
    """Fonction seuil : retourne 1 si x > 0, sinon 0."""
    return 1 if x > 0 else 0

def act_sigmoid(x):
    """Fonction sigmoïde : retourne une valeur entre 0 et 1."""
    return 1 / (1 + exp(-x))

def act_relu(x):
    """Fonction ReLU : retourne x si x > 0, sinon 0."""
    return max(0, x)



# Explications :

# La fonction sigmoïde prend une valeur réelle et la transforme en une valeur entre 0 et 1.
# Utilisée pour donner une non-linéarité au réseau de neurones.
# Elle est très utilisée dans les réseaux de neurones pour "activer" la sortie d’un neurone.
# Tu peux l’utiliser dans ton réseau en passant act_sigmoid comme fonction d’activation.

# Sortie:

# python3 main.py
# Input: [1.0, 2.0, 4.0]

# --- Test Neuron ---
# Neurone h1 (brut): 1.6
# Neurone h2 (brut): 0.5
# Neurone h1 (activé): 0.8320183851339245
# Neurone h2 (activé): 0.6224593312018546

# --- Test Layer ---
# Couche (valeurs brutes): [1.6, 0.5]
# Couche (valeurs activées): [0.8320183851339245, 0.6224593312018546]

# --- Test Network ---

# Sorties activées : [0.5003061329870411, 0.5126503661948724]

# Couche 1 (valeurs brutes): [1.6, 0.5]
# Couche 1 (valeurs activées): [0.8320183851339245, 0.6224593312018546]

# Couche 2 (valeurs brutes): [-0.12449186624037092, 0.3489837324807419, 0.02449186624037092]
# Couche 2 (valeurs activées): [0.46891716713519715, 0.5863711150170893, 0.506122660505889]

# Couche 3 (valeurs brutes): [0.0012245321011778165, 0.05061226605058891]
# Couche 3 (valeurs activées): [0.5003061329870411, 0.5126503661948724]


# Explications des résultats
# Test Neuron

# Neurone h1 (brut) et Neurone h2 (brut): sortie brute du calcul
# (produit des entrées et des poids + biais).
# Neurone h1 (activé) et Neurone h2 (activé): sortie après application de la
# fonction sigmoïde (valeur comprise entre 0 et 1).

# Test Layer

# Couche (valeurs brutes): sorties brutes de chaque neurone de la couche.
# Couche (valeurs activées): sorties activées (sigmoïde) de chaque neurone.

# Test Network

# Sorties activées: sorties finales du réseau, après passage dans toutes les couches et activation.
# Couche 1, 2, 3 (valeurs brutes/activées): affichage intermédiaire pour chaque couche du réseau,
# utile pour comprendre le fonctionnement étape par étape.

