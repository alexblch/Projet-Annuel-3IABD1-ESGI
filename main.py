#import image
import random
from PIL import Image

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


def perceptron( hidden_Layer, neurons, random, data, bias, size, lib):
    func = lib.perceptron
    func.restype = ctypes.c_double
    func.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int ,ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int)
    data_array = (ctypes.c_double * len(data))(*data)
    res = func(hidden_Layer, neurons, random, data_array, size, bias)
    return res



data = []
weight = []
print(weight)



n = int(input("Enter a number: "))
for i in range(n):
    data.append(i)
    weight.append(random.randint(-100, 100))
print(weight)
res = linear_model(data, weight, len(data), 1, lib)
print(f"res = {res}")

hidden_Layer = int(input("Enter a number of hidden layer: "))
neurons = int(input("Enter a number of neurons: "))
random = int(input("Enter a number of random: "))
res = perceptron(hidden_Layer, neurons, random, data, 1, len(data), lib)
print(f"res = {res}")
