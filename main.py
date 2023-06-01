#import image
import random
from PIL import Image
import numpy as np
import os

import ctypes

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
        img_list.append(img)
    return img_list

img_flatten_list = []
img_list = get_img_list("./Dataset")
for img in img_list:
    img_mat = get_matrice_image(img)
    img_flatten = img_mat.flatten()
    img_flatten_list.append(img_flatten)

for flatten in img_flatten_list:
    print(f'Image : {flatten}')


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


def perceptron( hidden_Layer, neurons, random, data, bias, size, lib):
    func = lib.perceptron
    func.restype = ctypes.c_double
    func.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int ,ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int)
    data_array = (ctypes.c_double * len(data))(*data)
    res = func(hidden_Layer, neurons, random, data_array, bias ,size)
    return res


rand = int(input("Enter a number of random: "))

data = []
weight = []

bias = 1

n = int(input("Enter a number: "))
for i in range(n):
    data.append(random.randint(0, 255))
    weight.append(random.choice(-random, random)
print(weight)
res = linear_model(data, weight, len(data), 1, lib)
print(f"res = {res}")

hidden_Layer = int(input("Enter a number of hidden layer: "))
neurons = int(input("Enter a number of neurons: "))
random = int(input("Enter a number of random: "))
res = perceptron(hidden_Layer, neurons, random, data, bias, len(data), lib)
print(f"res = {res}")
