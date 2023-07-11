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


def perceptron( hidden_Layer, neurons, random, data, bias, size, lib, nb_Class, prediction):
    func = lib.perceptron
    func.restype = ctypes.c_double
    func.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int ,ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
    data_array = (ctypes.c_double * len(data))(*data)
    prediction_array = (ctypes.c_int * len(prediction))(*prediction)
    res = func(hidden_Layer, neurons, random, data_array, bias ,size, nb_Class, prediction_array)
    return res

def get_file(hidden_Layer, neurons, random, size_image, lib, nb_Class):
    func = lib.create_file
    func.restype = None
    func.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    func(hidden_Layer, neurons, random, size_image, nb_Class)
    
def get_matrice_image(img):
    img_mat = np.array(img)
    return img_mat
    
img = Image.open("./dataset/terrain_de_football/train/foot (1).jpg")
img = img.resize((32,32))
img = img.convert('L')
flatten = get_matrice_image(img)
img_flatten = flatten.flatten()
print(img_flatten)
nb_Class = 3
hidden_Layer = int(input("hidden layer: "))
neurons = int(input("neurons: "))
random = int(input("random: "))
bias = int(input("bias: "))
get_file(hidden_Layer, neurons, random, 32*32, lib, nb_Class)
a = perceptron(hidden_Layer, neurons, random, img_flatten,bias, len(img_flatten), lib, nb_Class, [1,0,0])
print(a)