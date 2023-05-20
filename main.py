import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary('./libadd.so')

tanh = lib.tanh
tanh.argtypes = [ctypes.c_double]
tanh.restype = ctypes.c_double

n = 0.5

set_Data = lib.set_Data

set_Data.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
set_Data.restype = ctypes.POINTER(ctypes.c_double)


data = [255, 255, 255]
size = len(data)

array = (ctypes.c_double * size)(*data)
result = lib.set_Data(array, size)
result_list = [result[i] for i in range(size)]

print(f"vecteur setdata = {result_list}")

print(tanh(n))
