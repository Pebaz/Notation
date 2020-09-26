import random
from . nndt import NNDT, _Struct


class Tuple(_Struct, NNDT):
    pass

'''
@struct  # Returns Tuple[Int, String[10]]
class Car:
    NumWheels: Int
    Make: String[10]
'''
