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


    def get_flatten_img_list(path_data, largeur, hauteur):
        img_list = get_img_list(path_data, largeur, hauteur)
        img_flatten_list = []
        for img in img_list:
            img_mat = get_matrice_image(img)
            img_flatten = img_mat.flatten()
            img_flatten_list.append(img_flatten)
        return img_flatten_list


    def train_linear_model(img_list, class_size, img_size, className, training_iteration, my_lib):
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
                         ctypes.c_int, ctypes.c_int]
        func.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
        list_train = func(ctype_img_list, ctype_class_size, img_size, training_iteration)
        return list_train


    def test_linear_model(img_list, class_size, img_size, weight_list, my_lib):
        func = my_lib.test_linear_model
        ctype_img_list = (ctypes.POINTER(ctypes.c_int) * len(img_list))()
        for i in range(len(img_list)):
            sublist = img_list[i]
            sublist_arr = (ctypes.c_int * (len(sublist) + 1))(*sublist)
            ctype_img_list[i] = sublist_arr

        ctype_weight_list = (ctypes.POINTER(ctypes.c_float) * len(weight_list))()
        for i in range(len(weight_list)):
            sublist = weight_list[i]
            sublist_arr = sublist
            ctype_weight_list[i] = sublist_arr

        ctype_class_size = class_size.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        func.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                         ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                         ctypes.c_int]
        func.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
        list_train = func(ctype_img_list, ctype_class_size, ctype_weight_list, img_size)
        # list_train = [[4, 5, 6], [7, 8, 9]]
        return list_train


    start = time.time()
    largeur_img = 128
    hauteur_img = 128
    training_iteration = 150000
    foot_list_train = get_flatten_img_list(r"dataset\terrain_de_football\train", largeur_img, hauteur_img)
    basket_list_train = get_flatten_img_list(r"dataset\terrain_de_basket\train", largeur_img, hauteur_img)
    tennis_list_train = get_flatten_img_list(r"dataset\terrain_de_tennis\train", largeur_img, hauteur_img)

    img_list_train = foot_list_train + basket_list_train + tennis_list_train

    class_size_train = np.array([len(foot_list_train), len(basket_list_train), len(tennis_list_train)])

    result_train_football_train = train_linear_model(img_list_train, class_size_train, largeur_img * hauteur_img,
                                                     "football",training_iteration, my_lib)
    result_train_basket_train = train_linear_model(img_list_train, class_size_train, largeur_img * hauteur_img,
                                                   "basket", training_iteration,my_lib)
    result_train_tennis_train = train_linear_model(img_list_train, class_size_train, largeur_img * hauteur_img,
                                                   "tennis",training_iteration, my_lib)
    img_list_size_train = len(foot_list_train) + len(basket_list_train) + len(tennis_list_train)
    train_football = []
    success = 0
    for i in range(training_iteration):
        if result_train_football_train[1][i] == 1.0:
            success += 1
        train_football.append(success / (i+1) * 100)
    print("Traning Football Linear Modele Accuracy :", "%.2f" % (success / training_iteration * 100), "%")
    plt.plot(train_football)
    plt.xlabel('Itération')
    plt.ylabel('Success(%)')
    plt.title('Train Football Linear Model')
    plt.show()

    train_basket = []
    success = 0
    for i in range(training_iteration):
        if result_train_basket_train[1][i] == 1.0:
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
        if result_train_tennis_train[1][i] == 1.0:
            success += 1
        train_tennis.append(success / (i + 1) * 100)
    print("Traning Tennis Linear Modele Accuracy :", "%.2f" % (success / training_iteration * 100), "%")
    plt.plot(train_tennis)
    plt.xlabel('Itération')
    plt.ylabel('Success(%)')
    plt.title('Train Tennis Linear Model')
    plt.show()
    weight_list_train = [result_train_football_train[0], result_train_basket_train[0], result_train_tennis_train[0]]

    foot_list_test = get_flatten_img_list(r"dataset\terrain_de_football\test", largeur_img, hauteur_img)
    basket_list_test = get_flatten_img_list(r"dataset\terrain_de_basket\test", largeur_img, hauteur_img)
    tennis_list_test = get_flatten_img_list(r"dataset\terrain_de_tennis\test", largeur_img, hauteur_img)
    img_list_test = foot_list_test + basket_list_test + tennis_list_test

    ctype_img_list_test = (ctypes.POINTER(ctypes.c_int) * len(img_list_test))()

    class_size_test = np.array([len(foot_list_test), len(basket_list_test), len(tennis_list_test)])
    ctype_class_size_test = class_size_test.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    label_and_output = test_linear_model(img_list_test, class_size_test, largeur_img * largeur_img, weight_list_train,
                                         my_lib)
    img_list_size_test = len(foot_list_test) + len(basket_list_test) + len(tennis_list_test)
    success = 0
    test_result = []
    for i in range(img_list_size_test):
        if label_and_output[0][i] == label_and_output[1][i]:
            success += 1
        test_result.append(success / (i+1) * 100)
    plt.plot(test_result)
    plt.xlabel('Itération')
    plt.ylabel('Accuracy(%)')
    plt.title('Test Linear Model')
    plt.ylim(1,100)
    plt.show()
    print("Test Modele Accuracy :", "%.2f" % (success / img_list_size_test * 100), "%")
    print("Temps d'éxécution :", round(time.time() - start, 2),"secondes")

