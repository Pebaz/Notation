from . nndt import *
from . string import String
from . number import Int, Float
from . array import Array


# TODO(pebaz): Fix up these modules since they would be cool NNDTs to have.
# The main problem as of 10/2/20 is that this (below) returns (NaN, NaN, NaN):
'''
@nn
def normalize_vec(tup: Vec3) -> Tuple[Float, Float, Float]:
    x, y, z = tup
    length = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    return x / length, y / length, z / length


print()
print('Result  (1, 0, 1):', normalize_vec[Vec3(x=1, y=0, z=1)])
print('Predict (1, 0, 1):', normalize_vec(Vec3([1, 0, 1])))
print()
print('--------------------')
'''
# from . tuple_ import Tuple
# from . struct import struct
