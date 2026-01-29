import numpy as np
import matplotlib.pyplot as plt

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
