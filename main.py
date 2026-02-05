from squelette import *
from chargement_dataset_image import *

def test_init_classique():
    nb_x = 3
    nb_h = 4
    nb_y = 2
    params = initialisation_classique(nb_x, nb_h, nb_y)
    print("W1 =", params["W1"])
    print("b1 =", params["b1"])
    print("W2 =", params["W2"])
    print("b2 =", params["b2"])


def main():
    test_init_classique()

def reseau_2(X, Y, dimension_couches, learning_rate, epoch) :
    """
    Argument :
    X -- valeurs d'entrées (input)
    nb_x -- nombre de neurones input layer
    nb_h -- nombre de neurones hidden layer
    nb_y -- nombre de neurones output layer

    Return :
    pred -- valeur de sortie (output)
    """
    nb_x = dimension_couches[0]
    nb_h = dimension_couches[1]
    nb_y = dimension_couches[2]

    parametres = initialisation_classique(nb_x, nb_h, nb_y)

    W1 = parametres["W1"]
    b1 = parametres["b1"]
    W2 = parametres["W2"]
    b2 = parametres["b2"]

    gradients = {}
    couts = []

    for i in range (0, epoch) :

        # Forward prop : Lineaire + ReLu -> Lineaire + Sigmoid
        A1, cache1 = forward_propagation(X, W1, b1, "relu")
        A2, cache2 = forward_propagation(A1, W2, b2, "sigmoid")

        # Calcul du cout
        cout = fonction_cout(A2, Y)

        # Backward prop
        gradients = backward_propagation(cache1, cache2, Y, A2)

        # Changement des poids 
        parametres = maj_parametres(parametres, gradients, learning_rate)

        W1 = parametres["W1"]
        b1 = parametres["b1"]
        W2 = parametres["W2"]
        b2 = parametres["b2"]

        if i % 100 == 0 or i == epoch - 1 :
            print("Cout après l'epoch {}: {}".format(i, np.squeeze(cout)))
        if i % 100 == 0:
            couts.append(cout)

    return parametres, couts

def test_reseau_image(dossier_chat, dossier_chien, dimension_couches, learning_rate, epoch) :
    X, Y = charger_dataset(dossier_chat, dossier_chien)
    parametres_finaux, couts = reseau_2(X, Y, dimension_couches, learning_rate, epoch)
    return parametres_finaux

if __name__ == "__main__":
    # main()
    '''
    # ====================================================================================================
    # ======================================Test 1========================================================
    # ====================================================================================================
    # X : 4 exemples avec 2 caractéristiques
    X = np.array([[0, 0, 1, 1], 
                [0, 1, 0, 1]])

    # Y : Le résultat attendu
    Y = np.array([[0, 1, 1, 0]])

    # Dimensions : 2 entrées -> 15 neurones cachés -> 1 sortie
    dimension_couches = [2, 15, 1]

    print("----- Démarrage de l'entraînement -----")
    # On lance ton modèle !
    parametres_finaux, couts = reseau_2(X, Y, dimension_couches, learning_rate=1, epoch=5000)

    # On fait une prédiction finale pour voir si ça marche
    W1 = parametres_finaux["W1"]
    b1 = parametres_finaux["b1"]
    W2 = parametres_finaux["W2"]
    b2 = parametres_finaux["b2"]

    A1, _ = forward_propagation(X, W1, b1, "relu")
    A2_final, _ = forward_propagation(A1, W2, b2, "sigmoid")

    print("\nPrédictions finales du réseau de deux couches :")
    print(A2_final)
    print("Vraies étiquettes (Y) :")
    print(Y)

    # Affichage de la courbe de coût
    plt.plot(couts)
    plt.ylabel('Coût')
    plt.xlabel('Itérations (x100)')
    plt.title("Courbe d'apprentissage")
    plt.show()
    # ====================================================================================================
    # ======================================Test 1========================================================
    # ====================================================================================================
    '''

    # ====================================================================================================
    # ======================================Test 2========================================================
    # ====================================================================================================
    parametres_finaux = test_reseau_image(dossier_chat="animals/cat", dossier_chien="animals/dog", dimension_couches = [12288, 30, 1], learning_rate = 0.003, epoch = 5000)
    print(parametres_finaux)

    image_originale = Image.open("images.jpg")
    image_lisse = image_originale.resize((64, 64))
    image_array = np.array(image_lisse)
    mon_image_X = image_array.reshape((1, 12288)).T
    mon_image_X = mon_image_X / 255.0

    # Prediction
    W1 = parametres_finaux["W1"]
    b1 = parametres_finaux["b1"]
    W2 = parametres_finaux["W2"]
    b2 = parametres_finaux["b2"]

    Z1 = np.dot(W1, mon_image_X) + b1
    A1 = np.maximum(0, Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid

    # Résultat
    probabilite = A2[0][0]
    
    print("\n--- RÉSULTAT DU TEST ---")
    print(f"Probabilité calculée : {probabilite:.4f}")

    if probabilite > 0.5:
        print(f" C'est un chat ! (Confiance : {probabilite*100:.2f}%)")
    else:
        confiance_chien = (1 - probabilite) * 100
        print(f" C'est un chient ! (Confiance : {confiance_chien:.2f}%)")