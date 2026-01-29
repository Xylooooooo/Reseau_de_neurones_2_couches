import numpy as np
import matplotlib.pyplot as plt
from fonction_activation import sigmoid, relu

def initialisation_classique(nb_x, nb_h, nb_y) : 
    """
    Argument :
    nb_x -- taille de la couche d'entrée
    nb_h -- taille de la couche cachée
    nb_y -- taille de la couche de sortie

    Returns :
    parametres -- dictionnaire comportants les paramètres W1, b1, W2, b2
    """

    np.random.seed(1)

    parametres = {}
    parametres["W1"] = np.random.randn(nb_h, nb_x) * 0.01
    parametres["b1"] = np.zeros((nb_h, 1))
    parametres["W2"] = np.random.randn(nb_y, nb_h) * 0.01
    parametres["b2"] = np.zeros((nb_y, 1))

    return parametres

def forward_propagation(A_precedent, W, b, fct_activation) :
    """
    Argument :
    A_precedent -- activations de la couche précédente (ou les entrées du réseau)
    W -- poids de la couche courante
    b -- biais de la couche courante
    fct_activation -- fonction d'activation à utiliser ("sigmoid" ou "relu")

    Returns :
    A -- activations de la couche courante
    cache -- dictionnaire contenant "A_precedent", "W", "b" et "Z" pour la rétropropagation
    """

    Z = np.dot(W, A_precedent) + b

    if fct_activation == "sigmoid" :
        A = sigmoid(Z)
    elif fct_activation == "relu" :
        A = relu(Z)

    cache = {
        "A_precedent": A_precedent,
        "W": W,
        "b": b,
        "Z": Z
    }

    return A, cache