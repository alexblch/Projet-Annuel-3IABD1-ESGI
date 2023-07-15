#import image
import random
from PIL import Image
import numpy as np
import os

import ctypes

#library
lib = ctypes.cdll.LoadLibrary('./libadd.so')

def perceptron( hidden_Layer, neurons, random, data, bias, size, lib, nb_Class, prediction, learning_rate):
    func = lib.perceptron
    func.restype = ctypes.c_double
    func.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int ,ctypes.POINTER(ctypes.c_double), ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_double)
    data_array = (ctypes.c_double * len(data))(*data)
    prediction_array = (ctypes.c_int * len(prediction))(*prediction)
    res = func(hidden_Layer, neurons, random, data_array, bias ,size, nb_Class, prediction_array, learning_rate)
    return res

def get_file(hidden_Layer, neurons, random, size_image, lib, nb_Class):
    func = lib.create_file
    func.restype = None
    func.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    func(hidden_Layer, neurons, random, size_image, nb_Class)

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


nbClass = 3

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

epochs = 100000

data = []
weight = []
print(tennisTrain)
print(f'{len(tennisTrain)}')
bias = 1
learning_rate = -0.01
nb_Class = 3
hidden_Layer = int(input("Enter a number of hidden layer: "))
neurons = int(input("Enter a number of neurons: "))
rand = int(input("Enter a number of random: "))
bias = int(input("Enter a number of bias: "))
size = 32*32
result = []
value = 0
get_file(hidden_Layer, neurons, rand, size, lib, nb_Class)

for i in range(epochs):
    r = random.randint(0, len(footTrain)-1)
    if r == 0:
        b = random.randint(0, len(footTrain)-1)
        prediction = [1, 0, 0]
        value = perceptron(hidden_Layer, neurons, rand, footTrain[b], bias, size, lib, nb_Class, prediction, learning_rate)
    if r == 1:
        b = random.randint(0, len(tennisTrain)-1)
        prediction = [0, 1, 0]
        value = perceptron(hidden_Layer, neurons, rand, tennisTrain[b], bias, size, lib, nb_Class, prediction, learning_rate)
    if r == 2:
        b = random.randint(0, len(basketTrain)-1)
        prediction = [0, 0, 1]
        value = perceptron(hidden_Layer, neurons, rand, basketTrain[b], bias, size, lib, nb_Class, prediction, learning_rate)
    result.append(value)
    #with open("file/outclass.txt") as f:
    print(f'Epochs : {i} / {epochs}, result : {value}')

print(f'results : {result}')
    
    
    




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

