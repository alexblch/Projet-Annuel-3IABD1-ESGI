import ctypes
import random

lib = ctypes.cdll.LoadLibrary('./libadd.so')

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

def display_results(val1, val2, val3, val4):
    print("0,0: ", val1)
    print("0,1: ", val2)
    print("1,0: ", val3)
    print("1,1: ", val4)

nb_Class = 1
rand = int(input("Enter a number of random: "))
get_file(1, 2, rand, 2, lib, nb_Class)

learning_rate = 0.01
epoch = 20000
val1 = 0
val2 = 0
val3 = 0
val4 = 0
list = [[0, 0], [0, 255], [255, 0], [255, 255]]
#XOR
get_file(1, 2, rand, 2, lib, nb_Class)
for i in range(epoch):
    a = random.randint(0, 3)
    if a == 0:
        val1 = perceptron(1, 2, rand, [0.0,0.0], -1, 2, lib, nb_Class, [-1], learning_rate)
        if val1 <= 0:
            print(True)
        else:
            print(False)
    if a == 1:
        val2 = perceptron(1, 2, rand, [255.0,0], -1, 2, lib, nb_Class, [1], learning_rate)
        if val2 >= 0:
            print(True)
        else:
            print(False)
    if a == 2:
        val3 = perceptron(1, 2, rand, [0,255.0], -1, 2, lib, nb_Class, [1], learning_rate)
        if val3 >= 0:
            print(True)
        else:
            print(False)
    if a == 3:
        val4 = perceptron(1, 2, rand, [255.0,255.0], -1, 2, lib, nb_Class, [-1], learning_rate)
        if val4 <= -0:
            print(True)
        else:
            print(False)
    print(f'Epoch: {i}, [0, 0]: {val1}, [1, 0]: {val2}, [0, 1]: {val3}, [1, 1]: {val4}')

display_results(val1, val2, val3, val4)
