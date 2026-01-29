import numpy as np

def sigmoid(X) : 
    """
    Argument :
    X -- tableau numpy de n'importe quelle dimension

    Returns :
    Z -- le résultat de l'application de la fonction sigmoïde sur chaque élément de X
    """

    Z = 1 / (1 + np.exp(-X))

    return Z

def relu(X) :
    """
    Argument :
    X -- tableau numpy de n'importe quelle dimension

    Returns :
    Z -- le résultat de l'application de la fonction relu sur chaque élément de X
    """

    Z = np.maximum(0, X)

    return Z