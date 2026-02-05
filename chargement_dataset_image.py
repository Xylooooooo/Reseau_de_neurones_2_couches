import os
from PIL import Image
import numpy as np

def charger_dataset(dossier_chat, dossier_chien, taille_image=(64, 64)):
    """
    Charge les images, les redimensionne, les normalise et crée les labels
    """
    data = []
    labels = []
    
    # Chargement chat (Y=1)
    for fichier in os.listdir(dossier_chat) :
        chemin = os.path.join(dossier_chat, fichier)

        # Redimensionner
        img = Image.open(chemin).resize(taille_image)
        # Convertir en tableau numpy
        img_array = np.array(img)
        # Aplatir : De (64,64,3) vers (12288,)
        img_flatten = img_array.reshape(-1)

        data.append(img_flatten)
        labels.append(1)


    # Chargement chien (Y=0)
    for fichier in os.listdir(dossier_chien):
        chemin = os.path.join(dossier_chien, fichier)

        img = Image.open(chemin).resize(taille_image)
        img_array = np.array(img)
        img_flatten = img_array.reshape(-1)
        
        data.append(img_flatten)
        labels.append(0)

            
    # Pour l'instant data est une liste de taille (m, n_x) - m = nb d'exemples, n_x = nb de caractéristiques (12288 pour 64x64x3)
    # Le réseau attend (n_x, m), donc on transpose
    X = np.array(data).T  
    Y = np.array(labels).reshape(1, -1)
    
    # On met les valeurs entre 0 et 1
    X = X / 255.0
    
    # On mélange les colonnes de X et Y ensemble
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    X = X[:, permutation]
    Y = Y[:, permutation]
    
    print(f"Terminé ! Taille de X : {X.shape}, Taille de Y : {Y.shape}")
    return X, Y