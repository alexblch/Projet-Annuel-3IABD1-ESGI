#import image
import random
from PIL import Image
import numpy as np
import os

import ctypes

#library
lib = ctypes.cdll.LoadLibrary('./libadd.so')

def linear_model(data, weight, size, bias, lib):
    func = lib.linear_model
    func.restype = ctypes.c_double
    func.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int)
    data_array = (ctypes.c_double * len(data))(*data)
    weight_array = (ctypes.c_double * len(weight))(*weight)
    res = func(data_array, weight_array, size, bias)
    return res


def perceptron( hidden_Layer, neurons, random, data, bias, size, lib, nb_Class, prediction, learning_rate):
    func = lib.perceptron
    func.restype = ctypes.c_double
    func.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int ,ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_double)
    data_array = (ctypes.c_double * len(data))(*data)
    prediction_array = (ctypes.c_int * len(prediction))(*prediction)
    res = func(hidden_Layer, neurons, random, data_array, bias ,size, nb_Class, prediction_array, learning_rate)
    return res

def get_file(hidden_Layer, neurons, random, size_image, lib, nb_Class):
    func = lib.create_file
    func.restype = None
    func.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    func(hidden_Layer, neurons, random, size_image, nb_Class)

nb_Class = 1
rand = int(input("Enter a number of random: "))
get_file(1, 2, rand, 2, lib, nb_Class)

learning_rate = 0.15

#XOR
for i in range(1000):
    val = perceptron(1, 2, 2, [0.0,0.0], 0, 2, lib, nb_Class, [-1], learning_rate)
    if(i == 999):
        print("XOR for [0, 0] = ", val)
    val = perceptron(1, 2, 2, [255.0,0], 0, 2, lib, nb_Class, [1], learning_rate)
    if(i == 999):
        print("XOR for [1, 0] = ", val)
    val = perceptron(1, 2, 2, [0.0,255.0], 0, 2, lib, nb_Class, [1], learning_rate)
    if(i == 999):
        print("XOR for [0, 1] = ", val)
    val = perceptron(1, 2, 2, [255.0,255.0], 0, 2, lib, nb_Class, [-1], learning_rate)
    if(i == 999):
        print("XOR for [1, 1] = ", val)
    print("Iteration ", i, " done")

nb_Class = 3
# Convertie l'image en matrice
def get_matrice_image(img):
    img_mat = np.array(img)
    return img_mat

# Retourne la liste d'image d'un dossier
def get_img_list(directory):
    img_dir = directory
    img_list = []
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = Image.open(img_path)
        img_resized = img.resize((32,32))
        img_black_white = img_resized.convert('L')
        img_list.append(img_black_white)
    return img_list

#get images
footTrain = []
footTest = []

tennisTrain = []
tennisTest = []

basketTrain = []
basketTest = []

img_list = get_img_list("./dataset/terrain_de_football/train/")
for img in img_list:
    img_mat = get_matrice_image(img)
    img_flatten = img_mat.flatten()
    footTrain.append(img_flatten)
    
img_list = get_img_list("./dataset/terrain_de_football/test/")
for img in img_list:
    img_mat = get_matrice_image(img)
    img_flatten = img_mat.flatten()
    footTest.append(img_flatten)
    
    
img_list = get_img_list("./dataset/terrain_de_tennis/train/")
for img in img_list:
    img_mat = get_matrice_image(img)
    img_flatten = img_mat.flatten()
    tennisTrain.append(img_flatten)
    
img_list = get_img_list("./dataset/terrain_de_tennis/test/")
for img in img_list:
    img_mat = get_matrice_image(img)
    img_flatten = img_mat.flatten()
    tennisTest.append(img_flatten)
    
img_list = get_img_list("./dataset/terrain_de_basket/train/")
for img in img_list:
    img_mat = get_matrice_image(img)
    img_flatten = img_mat.flatten()
    basketTrain.append(img_flatten)

img_list = get_img_list("./dataset/terrain_de_basket/test/")
for img in img_list:
    img_mat = get_matrice_image(img)
    img_flatten = img_mat.flatten()
    basketTest.append(img_flatten)




#results of linear model

football_train = []
football_test = []

tennis_train = []
tennis_test = []

basket_train = []
basket_test = []


#results of perceptron
football_trainMLP = []
football_testMLP = []

tennis_trainMLP = []
tennis_testMLP = []

basket_trainMLP = []
basket_testMLP = []

#print(f'football train : \n{footTrain}')



data = []
weight = []

bias = 1

"""for flatten in footTrain:
    for i in range(len(flatten)):
        weight.append(random.randint(-rand, rand))
    football_train.append(linear_model(flatten, weight, len(flatten), bias, lib))
    weight = []
for flatten in footTest:
    for i in range(len(flatten)):
        weight.append(random.randint(-rand, rand))
    football_test.append(linear_model(flatten, weight, len(flatten), bias, lib))
    weight = []

for flatten in tennisTrain:
    for i in range(len(flatten)):
        weight.append(random.randint(-rand, rand))
    tennis_train.append(linear_model(flatten, weight, len(flatten), bias, lib))
    weight = []
for flatten in tennisTest:
    for i in range(len(flatten)):
        weight.append(random.randint(-rand, rand))
    tennis_test.append(linear_model(flatten, weight, len(flatten), bias, lib))
    weight = []
    
for flatten in basketTrain:
    for i in range(len(flatten)):
        weight.append(random.randint(-rand, rand))
    basket_train.append(linear_model(flatten, weight, len(flatten), bias, lib))
    weight = []
for flatten in basketTest:
    for i in range(len(flatten)):
        weight.append(random.randint(-rand, rand))
    basket_test.append(linear_model(flatten, weight, len(flatten), bias, lib))
    weight = []
    

print(f'Result learning foot :\n{football_train}\n')
print(f'Result test foot :\n{football_test}\n\n')

print(f'Result learning tennis :\n{tennis_train}\n')
print(f'Result test tennis :\n{tennis_test}\n\n')

print(f'Result learning basket :\n{basket_train}\n')
print(f'Result test basket :\n{basket_test}\n\n')"""

hidden_Layer = int(input("Enter a number of hidden layer: "))
neurons = int(input("Enter a number of neurons: "))
random = int(input("Enter a number of random: "))
bias = int(input("Enter a number of bias: "))
size = 32*32

get_file(hidden_Layer, neurons, random, size, lib, nb_Class)
print(footTrain)
for flatten in footTrain:
    football_trainMLP.append(perceptron(hidden_Layer, neurons, random, flatten, bias, len(flatten), lib, nb_Class, [1,-1,-1], learning_rate))
for flatten in footTest:
    football_testMLP.append(perceptron(hidden_Layer, neurons, random, flatten, bias, len(flatten), lib, nb_Class, [1,-1,-1], learning_rate))
    
for flatten in tennisTrain:
    tennis_trainMLP.append(perceptron(hidden_Layer, neurons, random, flatten, bias, len(flatten), lib, nb_Class, [-1,1,-1], learning_rate))
for flatten in tennisTest:
    tennis_testMLP.append(perceptron(hidden_Layer, neurons, random, flatten, bias, len(flatten), lib, nb_Class, [-1,1,-1], learning_rate))
    
for flatten in basketTrain:
    basket_trainMLP.append(perceptron(hidden_Layer, neurons, random, flatten, bias, len(flatten), lib, nb_Class, [-1,-1,1], learning_rate))
for flatten in basketTest:
    basket_testMLP.append(perceptron(hidden_Layer, neurons, random, flatten, bias, len(flatten), lib, nb_Class, [-1,-1,1], learning_rate))
    

print(f'Result learning foot for perceptron :\n{football_trainMLP}\n')
print(f'Result test foot for perceptron :{football_testMLP}\n\n')

print(f'Result learning tennis for perceptron :\n{tennis_trainMLP}\n')
print(f'Result test tennis for perceptron :\n{tennis_testMLP}\n\n')

print(f'Result learning basket for perceptron :\n{basket_trainMLP}\n')
print(f'Result test basket for perceptron :\n{basket_testMLP}\n\n')


#plotting
import matplotlib.pyplot as plt
# Graphique pour le football




# Graphique pour le football
plt.figure(1)
plt.scatter(range(len(football_trainMLP)), football_trainMLP, label='Entraînement', color='red')
plt.scatter(range(len(football_testMLP)), football_testMLP, label='Test', color='blue')
plt.xlabel('Échantillons')
plt.ylabel('Résultats')
plt.title('FootballMLP')
plt.savefig('./graph/footballMLP.png')
plt.legend()

# Graphique pour le tennis
plt.figure(2)
plt.scatter(range(len(tennis_trainMLP)), tennis_trainMLP, label='Entraînement', color='red')
plt.scatter(range(len(tennis_testMLP)), tennis_testMLP, label='Test', color='blue')
plt.xlabel('Échantillons')
plt.ylabel('Résultats')
plt.title('TennisMLP')
plt.savefig('./graph/tennisMLP.png')
plt.legend()

# Graphique pour le basket
plt.figure(3)
plt.scatter(range(len(basket_trainMLP)), basket_trainMLP, label='Entraînement', color='red')
plt.scatter(range(len(basket_testMLP)), basket_testMLP, label='Test', color='blue')
plt.xlabel('Échantillons')
plt.ylabel('Résultats')
plt.title('BasketMLP')
plt.savefig('./graph/basketMLP.png')
plt.legend()

# Afficher tous les graphiques
plt.show()

