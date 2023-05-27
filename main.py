#import image
from PIL import Image

import ctypes

#library
lib = ctypes.cdll.LoadLibrary('./libadd.so')


def tanh(x, lib):
    func = lib.tanh
    func.restype = ctypes.c_double
    func.argtypes = [ctypes.c_double]
    return func(x)

def linear_model(data, weight, size, bias, lib):
    func = lib.linear_model
    func.restype = ctypes.c_double
    func.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int)
    data_array = (ctypes.c_double * len(data))(*data)
    weight_array = (ctypes.c_double * len(weight))(*weight)
    res = func(data_array, weight_array, size, bias)
    return res



data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
print(weight)

res : float

n = int(input("Enter a number: "))
print(tanh(n, lib))
res = linear_model(data, weight, len(data), 1, lib)
print(f"res = {res}")
res = tanh(res, lib)
print(f'result = {res}')

