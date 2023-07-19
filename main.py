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

def propagate(hidden_Layer, neurons, random, data, bias, size, lib, nb_Class):
    func = lib.propagate
    func.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
                     ctypes.c_double, ctypes.c_int, ctypes.c_int)
    func.restype = ctypes.POINTER(ctypes.c_double)
    
    data_array = (ctypes.c_double * len(data))(*data)
    res = func(hidden_Layer, neurons, random, data_array, bias, size, nb_Class)
    
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
        img_resized = img.resize((25, 25))
        img_float = np.array(img_resized, dtype=float) / 255.0 # convert image to numpy array and change data type to float
        img_list.append(img_float)
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
epochs = 10

data = []
weight = []
bias = 10
learning_rate = -0.05
nb_Class = 3
hidden_Layer = int(input("Enter a number of hidden layer: "))
neurons = int(input("Enter a number of neurons: "))
rand = int(input("Enter a number of random: "))
bias = int(input("Enter a number of bias: "))
size = 25*25*3
result = []
value = 0
get_file(hidden_Layer, neurons, rand, size, lib, nb_Class)

"""for i in range(epochs):
    r = random.randint(0, 2)
    if r == 0:
        b = random.randint(0, len(footTrain)-1)
        prediction = [1, -1, -1]
        value = perceptron(hidden_Layer, neurons, rand, footTrain[b], bias, size, lib, nb_Class, prediction, learning_rate)
        print(f'epoch : {i}, value : {value}')
    if r == 1:
        b = random.randint(0, len(tennisTrain)-1)
        prediction = [-1, 1, -1]
        value = perceptron(hidden_Layer, neurons, rand, tennisTrain[b], bias, size, lib, nb_Class, prediction, learning_rate)
        print(f'epoch : {i}, value : {value}')
    if r == 2:
        b = random.randint(0, len(basketTrain)-1)
        prediction = [-1, -1, 1]
        value = perceptron(hidden_Layer, neurons, rand, basketTrain[b], bias, size, lib, nb_Class, prediction, learning_rate)
        print(f'epoch : {i}, value : {value}')
    result.append(value)"""
    
    
    
for i in range(epochs):
    r = random.randint(0, 2)
    if r == 0:
        prediction = [1, -1, -1]
        for flatten in footTrain:
            value = perceptron(hidden_Layer, neurons, rand, flatten, bias, size, lib, nb_Class, prediction, learning_rate)
            result.append(prediction[0] - value)
    if r == 1:
        prediction = [-1, 1, -1]
        for flatten in tennisTrain:
            value = perceptron(hidden_Layer, neurons, rand, flatten, bias, size, lib, nb_Class, prediction, learning_rate)
            result.append(prediction[0] - value)
    if r == 2:
        prediction = [-1, -1, 1]
        for flatten in basketTrain:
            value = perceptron(hidden_Layer, neurons, rand, flatten, bias, size, lib, nb_Class, prediction, learning_rate)
            result.append(prediction[0] - value)
    
    #with open("file/outclass.txt") as f:
    print(f'epoch : {i}, value : {value}')

prediction = [1, -1, -1]
accuracy = 0    
l = []
total = len(footTest) + len(tennisTest) + len(basketTest)
for i in range(len(footTest)):
    index = 0
    value = propagate(hidden_Layer, neurons, rand, footTest[i], bias, size, lib, nb_Class)#perceptron(hidden_Layer, neurons, rand, footTest[i], bias, size, lib, nb_Class, prediction, learning_rate)
    with open("file/outclass.txt") as f:
        contenu = f.read().split(" ")
        new_contenu = []  # create a new list for the converted values
        for j in range(len(contenu)):
            if contenu[j] != '':  # only convert non-empty strings
                try:
                    new_contenu.append(float(contenu[j]))  # add the converted value to the new list
                except ValueError:
                    continue
        l.append(new_contenu)
        print(f'new_contenu : {new_contenu}')
        if new_contenu[index] == max(new_contenu):
            accuracy += 1

prediction = [-1, 1, -1]     
for i in range(len(tennisTest)):
    index = 1
    value = propagate(hidden_Layer, neurons, rand, tennisTest[i], bias, size, lib, nb_Class)#perceptron(hidden_Layer, neurons, rand, tennisTest[i], bias, size, lib, nb_Class, prediction, learning_rate)
    with open("file/outclass.txt") as f:
        contenu = f.read().split(" ")
        new_contenu = []  # create a new list for the converted values
        for j in range(len(contenu)):
            if contenu[j] != '':  # only convert non-empty strings
                try:
                    new_contenu.append(float(contenu[j]))  # add the converted value to the new list
                except ValueError:
                    continue
        l.append(new_contenu)
        print(f'new_contenu : {new_contenu}')
        if new_contenu[index] == max(new_contenu):
            accuracy += 1
prediction = [-1, -1, 1]            
for i in range(len(basketTest)):
    index = 2
    value = propagate(hidden_Layer, neurons, rand, basketTest[i], bias, size, lib, nb_Class)#perceptron(hidden_Layer, neurons, rand, basketTest[i], bias, size, lib, nb_Class, prediction, learning_rate)
    with open("file/outclass.txt") as f:
        contenu = f.read().split(" ")
        new_contenu = []  # create a new list for the converted values
        for j in range(len(contenu)):
            if contenu[j] != '':  # only convert non-empty strings
                try:
                    new_contenu.append(float(contenu[j]))  # add the converted value to the new list
                except ValueError:
                    continue
        l.append(new_contenu)
        print(f'new_contenu : {new_contenu}')
        if new_contenu[index] == max(new_contenu):
            accuracy += 1
            
print(f'Accuracy : {accuracy} / {total} = {accuracy/total}')
            
    
    
    




#plotting
import matplotlib.pyplot as plt


plt.figure()

# Trace la courbe
plt.plot(result)

# Donne un titre à la courbe
plt.title('Courbe du loss pour le foot dans le MLP training')

# Donne un nom à l'axe des x
plt.xlabel('Index')

# Donne un nom à l'axe des y
plt.ylabel('Valeur')

# Affiche la figure
plt.show()
plt.savefig('./graph/lossFootMLP.png')

