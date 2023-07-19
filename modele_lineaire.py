import ctypes
import time
import numpy as np
import os

from PIL import Image

import matplotlib.pyplot as plt

if __name__ == "__main__":
    chemin = "./lib2023_3A_IABD1_Demo_Interop_Cpp.dll"

    my_lib = ctypes.CDLL(
        "C:/Users/desan/Documents/ESGI/PA/PALibrary/cmake-build-debug/PALibrary.dll")


    # Convertie l'image en matrice
    def get_matrice_image(img):
        img_mat = np.array(img)
        return img_mat


    # Retourne la liste d'image d'un dossier
    def get_img_list(directory, largeur, hauteur):
        img_dir = directory
        img_list = []
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path)
            img_resize = img.resize((largeur, hauteur))
            img_black_and_white = img_resize.convert("L")
            img_list.append(img_black_and_white)
        return img_list


    def get_list_img_flatten(path_data, largeur, hauteur):
        img_list = get_img_list(path_data, largeur, hauteur)
        img_flatten_list = []
        for img in img_list:
            img_mat = get_matrice_image(img)
            img_flatten = img_mat.flatten()
            img_flatten_list.append(img_flatten)
        return img_flatten_list


    def train_linear_model(img_list, class_size, img_size, className, training_iteration, my_lib, modele_file_path):
        if className == "football":
            func = my_lib.train_linear_model_football
        elif className == "basket":
            func = my_lib.train_linear_model_basket
        else:
            func = my_lib.train_linear_model_tennis
        ctype_img_list = (ctypes.POINTER(ctypes.c_int) * len(img_list))()
        for i in range(len(img_list)):
            sublist = img_list[i]
            sublist_arr = (ctypes.c_int * (len(sublist) + 1))(*sublist)
            ctype_img_list[i] = sublist_arr
        ctype_class_size = class_size.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        func.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                         ctypes.POINTER(ctypes.c_int),
                         ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        func.restype = ctypes.POINTER(ctypes.c_int)
        list_train = func(ctype_img_list, ctype_class_size, img_size, training_iteration, modele_file_path.encode())
        return list_train


    def test_linear_model(img_list, class_size, img_size, path_file_model_football, path_file_model_basket,
                          path_file_model_tennis, my_lib):
        func = my_lib.test_linear_model
        ctype_img_list = (ctypes.POINTER(ctypes.c_int) * len(img_list))()
        for i in range(len(img_list)):
            sublist = img_list[i]
            sublist_arr = (ctypes.c_int * (len(sublist) + 1))(*sublist)
            ctype_img_list[i] = sublist_arr

        ctype_class_size = (ctypes.c_int * len(class_size))(*class_size)
        func.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                         ctypes.POINTER(ctypes.c_int),
                         ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        func.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
        list_train = func(ctype_img_list, ctype_class_size, img_size, path_file_model_football.encode(),
                          path_file_model_basket.encode(), path_file_model_tennis.encode())
        # list_train = [[4, 5, 6], [7, 8, 9]]
        return list_train


    def test_linear_model(img_list, class_size, img_size, path_file_model_football, path_file_model_basket,
                          path_file_model_tennis, my_lib):
        func = my_lib.test_linear_model
        ctype_img_list = (ctypes.POINTER(ctypes.c_int) * len(img_list))()
        for i in range(len(img_list)):
            sublist = img_list[i]
            sublist_arr = (ctypes.c_int * (len(sublist) + 1))(*sublist)
            ctype_img_list[i] = sublist_arr

        ctype_class_size = (ctypes.c_int * len(class_size))(*class_size)
        func.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                         ctypes.POINTER(ctypes.c_int),
                         ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        func.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
        list_train = func(ctype_img_list, ctype_class_size, img_size, path_file_model_football.encode(),
                          path_file_model_basket.encode(), path_file_model_tennis.encode())
        # list_train = [[4, 5, 6], [7, 8, 9]]
        return list_train


    def predict_class_img(img, img_size, path_file_model_football, path_file_model_basket,
                          path_file_model_tennis, my_lib):
        func = my_lib.predict_class
        """ctype_img = img.ctypes.data_as(ctypes.POINTER(ctypes.c_int))"""
        ctype_img = (ctypes.c_int * len(img))(*img)
        """for i in range(len(img)):
            print("PIXELS", i, "=", ctype_img[i])"""
        func.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
                         ctypes.c_char_p]
        func.restype = ctypes.c_int
        list_train = func(ctype_img, img_size, path_file_model_football.encode(),
                          path_file_model_basket.encode(), path_file_model_tennis.encode())
        return list_train


    start = time.time()
    largeur_img = 70
    hauteur_img = 70
    training_iteration = 100000

    foot_list_train = get_list_img_flatten(r"dataset\terrain_de_football\train", largeur_img, hauteur_img)
    basket_list_train = get_list_img_flatten(r"dataset\terrain_de_basket\train", largeur_img, hauteur_img)
    tennis_list_train = get_list_img_flatten(r"dataset\terrain_de_tennis\train", largeur_img, hauteur_img)

    
    foot_list_test = get_list_img_flatten(r"dataset\terrain_de_football\test", largeur_img, hauteur_img)
    basket_list_test = get_list_img_flatten(r"dataset\terrain_de_basket\test", largeur_img, hauteur_img)
    tennis_list_test = get_list_img_flatten(r"dataset\terrain_de_tennis\test", largeur_img, hauteur_img)

    
    #IMPORTANT mettre les chemins absolus des fichiers
    path_file_model_football = "chemin_absolu/football_model.txt"
    path_file_model_basket = "chemin_absolu/basket_model.txt"
    path_file_model_tennis = "chemin_absolu/tennis_model.txt"

    new_image = Image.open(r"dataset\terrain_de_basket/temp/3531_image1_1.jpg").resize(
        (largeur_img, hauteur_img)).convert("L")
    new_img_flatten = get_matrice_image(new_image).flatten()

    img_list_train = foot_list_train + basket_list_train + tennis_list_train

    class_size_train = np.array([len(foot_list_train), len(basket_list_train), len(tennis_list_train)])

    result_train_football_train = train_linear_model(img_list_train, class_size_train, largeur_img * hauteur_img,
                                                     "football", training_iteration, my_lib, path_file_model_football)
    result_train_basket_train = train_linear_model(img_list_train, class_size_train, largeur_img * hauteur_img,
                                                   "basket", training_iteration, my_lib, path_file_model_basket)
    result_train_tennis_train = train_linear_model(img_list_train, class_size_train, largeur_img * hauteur_img,
                                                   "tennis", training_iteration, my_lib, path_file_model_tennis)
    img_list_size_train = len(foot_list_train) + len(basket_list_train) + len(tennis_list_train)
    train_football = []
    success = 0
    for i in range(training_iteration):
        if result_train_football_train[i] == 1.0:
            success += 1
        train_football.append(success / (i + 1) * 100)
    print("Traning Football Linear Modele Accuracy :", "%.2f" % (success / training_iteration * 100), "%")
    plt.plot(train_football)
    plt.xlabel('Itération')
    plt.ylabel('Success(%)')
    plt.title('Train Football Linear Model')
    plt.show()

    train_basket = []
    success = 0
    for i in range(training_iteration):
        if result_train_basket_train[i] == 1.0:
            success += 1
        train_basket.append(success / (i + 1) * 100)
    print("Traning Basket Linear Modele Accuracy :", "%.2f" % (success / training_iteration * 100), "%")
    plt.plot(train_basket)
    plt.xlabel('Itération')
    plt.ylabel('Success(%)')
    plt.title('Train Basket Linear Model')
    plt.show()

    train_tennis = []
    success = 0
    for i in range(training_iteration):
        if result_train_tennis_train[i] == 1.0:
            success += 1
        train_tennis.append(success / (i + 1) * 100)
    print("Traning Tennis Linear Modele Accuracy :", "%.2f" % (success / training_iteration * 100), "%")
    plt.plot(train_tennis)
    plt.xlabel('Itération')
    plt.ylabel('Success(%)')
    plt.title('Train Tennis Linear Model')
    plt.show()
    weight_list_train = [result_train_football_train[0], result_train_basket_train[0], result_train_tennis_train[0]]

    img_list_test = foot_list_test + basket_list_test + tennis_list_test

    ctype_img_list_test = (ctypes.POINTER(ctypes.c_int) * len(img_list_test))()

    class_size_test = np.array([len(foot_list_test), len(basket_list_test), len(tennis_list_test)])
    ctype_class_size_test = class_size_test.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    label_and_output = test_linear_model(img_list_test, class_size_test, largeur_img * hauteur_img,
                                         path_file_model_football, path_file_model_basket, path_file_model_tennis,
                                         my_lib)
    img_list_size_test = len(foot_list_test) + len(basket_list_test) + len(tennis_list_test)

    football_barre_foot_model = 0
    basket_barre_foot_model = 0
    tennis_barre_foot_model = 0

    football_barre_basket_model = 0
    basket_barre_basket_model = 0
    tennis_barre_basket_model = 0

    football_barre_tennis_model = 0
    basket_barre_tennis_model = 0
    tennis_barre_tennis_model = 0

    for i in range(img_list_size_test):
        if label_and_output[0][i] == 1:
            if label_and_output[1][i] == 1:
                football_barre_foot_model = football_barre_foot_model + 1
            elif label_and_output[1][i] == 2:
                basket_barre_foot_model = basket_barre_foot_model + 1
            elif label_and_output[1][i] == 3:
                tennis_barre_foot_model = tennis_barre_foot_model + 1
        elif label_and_output[0][i] == 2:
            if label_and_output[1][i] == 1:
                football_barre_basket_model = football_barre_basket_model + 1
            elif label_and_output[1][i] == 2:
                basket_barre_basket_model = basket_barre_basket_model + 1
            elif label_and_output[1][i] == 3:
                tennis_barre_basket_model = tennis_barre_basket_model + 1
        elif label_and_output[0][i] == 3:
            if label_and_output[1][i] == 1:
                football_barre_tennis_model = football_barre_tennis_model + 1
            elif label_and_output[1][i] == 2:
                basket_barre_tennis_model = basket_barre_tennis_model + 1
            elif label_and_output[1][i] == 3:
                tennis_barre_tennis_model = tennis_barre_tennis_model + 1

    x_barre = [1,2,3]
    barre_foot_img = [football_barre_foot_model, basket_barre_foot_model, tennis_barre_foot_model]
    # Création du graphique à barres
    plt.bar(x_barre, barre_foot_img)
    # Personnalisation du graphique
    plt.xlabel('Classe prédit par le modèle')  # Étiquette de l'axe x
    plt.ylabel('Nombre de fois')  # Étiquette de l'axe y
    plt.title('Test Modèle Footbal Image')  # Titre du graphique
    plt.xticks(x_barre, ['Football', 'Basket', 'Tennis'])  # Étiquettes de l'axe x
    plt.yticks(range(0, max(barre_foot_img) + 1, 5))  # Échelle de l'axe y

    # Affichage du graphique
    plt.show()

    barre_basket_img = [football_barre_basket_model, basket_barre_basket_model, tennis_barre_basket_model]
    # Création du graphique à barres
    plt.bar(x_barre, barre_basket_img)
    # Personnalisation du graphique
    plt.xlabel('Classe prédit par le modèle')  # Étiquette de l'axe x
    plt.ylabel('Nombre de fois')  # Étiquette de l'axe y
    plt.title('Test Modèle Basket Image')  # Titre du graphique
    plt.xticks(x_barre, ['Football', 'Basket', 'Tennis'])  # Étiquettes de l'axe x
    plt.yticks(range(0, max(barre_basket_img) + 1, 5))  # Échelle de l'axe y

    # Affichage du graphique
    plt.show()

    barre_tennis_img = [football_barre_tennis_model, basket_barre_tennis_model, tennis_barre_tennis_model]
    # Création du graphique à barres
    plt.bar(x_barre, barre_tennis_img)
    # Personnalisation du graphique
    plt.xlabel('Classe prédit par le modèle')  # Étiquette de l'axe x
    plt.ylabel('Nombre de fois')  # Étiquette de l'axe y
    plt.title('Test Modèle Tennis Image')  # Titre du graphique
    plt.xticks(x_barre, ['Football', 'Basket', 'Tennis'])  # Étiquettes de l'axe x
    plt.yticks(range(0, max(barre_tennis_img) + 1, 5))  # Échelle de l'axe y

    # Affichage du graphique
    plt.show()

    success = 0
    test_result = []
    for i in range(img_list_size_test):
        if label_and_output[0][i] == label_and_output[1][i]:
            success += 1
        test_result.append(success / (i + 1) * 100)
    """plt.plot(test_result)
    plt.xlabel('Itération')
    plt.ylabel('Accuracy(%)')
    plt.title('Test Linear Model')
    plt.ylim(1, 100)
    plt.show()"""
    print("Test Modele Accuracy :", "%.2f" % (success / img_list_size_test * 100), "%")





    """for pixel in new_img_flatten:
        print (pixel)

    print("l * l : ", largeur_img * hauteur_img)"""


    prediction = predict_class_img(new_img_flatten, largeur_img * hauteur_img, path_file_model_football,
                                   path_file_model_basket, path_file_model_tennis, my_lib)

    if prediction == 1:
        print("Terrain de foot")
    elif prediction == 2:
        print("Terrain de basket")
    elif prediction == 3:
        print("Terrain de tennis")
    else:
        print("PB prediction =", prediction)

    print("Temps d'éxécution :", round(time.time() - start, 2), "secondes")