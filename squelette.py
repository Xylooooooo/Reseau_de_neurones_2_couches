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
        "Z": Z,
    }

    return A, cache

def fonction_cout(Y_pred, Y) :
    """
    Argument :
    Y_pred -- prédictions du modèle (activations de la couche de sortie)
    Y -- vraies étiquettes

    Returns :
    cout -- valeur de la fonction de coût
    """

    m = Y.shape[1]

    cout = -(1/m) * np.sum(Y * np.log(Y_pred) + (1 - Y) * np.log(1 - Y_pred))

    return cout

def backward(dZ, cache) :
    """
    Argument :
    dZ -- gradient de la fonction de coût par rapport à Z couche l
    cache -- dictionnaire contenant "A_precedent", "W", "b" et "Z" de la couche l

    Returns :
    dA_precedent -- gradient de la fonction de coût par rapport à A_precedent
    dW -- gradient de la fonction de coût par rapport à W
    db -- gradient de la fonction de coût par rapport à b
    """

    A_precedent = cache["A_precedent"]
    W = cache["W"]
    m = A_precedent.shape[1]

    dW = (1/m) * np.dot(dZ, A_precedent.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_precedent = np.dot(W.T, dZ)

    return dA_precedent, dW, db

def backward_propagation(cache1, cache2, Y, A2) :
    """
    Argument :
    cache1 -- dictionnaire contenant "A_precedent", "W", "b" et "Z" de la couche 1
    cache2 -- dictionnaire contenant "A_precedent", "W", "b" et "Z" de la couche 2
    Y -- vraies étiquettes
    A2 -- activations de la couche de sortie

    Returns :
    gradients -- dictionnaire contenant les gradients dA1, dW1, db1, dW2, db2
    """

    m = Y.shape[1]

    dZ2 = A2 - Y
    dA1, dW2, db2 = backward(dZ2, cache2)

    Z1 = cache1["Z"]
    dZ1 = dA1.copy()
    dZ1[Z1 <= 0] = 0
    dA0, dW1, db1 = backward(dZ1, cache1)

    gradients = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2,
    }

    return gradients

def maj_parametres(parametres,gradients, learning_rate) :
    """
    Argument :
    parametres -- dico comportant paramètres des deux couches
    gradients --  dico comportant les gradients
    learning_rate -- ..
    
    returns :
    parametres -- paramètres mis à jour
    """
    parametres["W1"] -= learning_rate * gradients["dW1"]
    parametres["b1"] -= learning_rate * gradients["db1"]
    parametres["W2"] -= learning_rate * gradients["dW2"]
    parametres["b2"] -= learning_rate * gradients["db2"]

    return parametres