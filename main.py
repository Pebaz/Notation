import math
from nndt import *

# @nn
# def append(array: Array[1, Int], num: Int) -> Array[2, Int]:
#     return array + [num]


# print()
# print('Result  ([1, 2]):', append[[1], 2])
# print('Predict ([1, 2]):', append([1], 2))
# print()
# print('--------------------')


# @nn
# def negate(number: Int) -> Int:
#     return -number


# print()
# print('Result  (-123):', negate[123])
# print('Predict (-124):', negate(124))
# print()
# print('--------------------')


# @nn
# def add(a: Int, b: Int) -> Int:
#     return a + b

# print()
# print('Result  (300):', add[100, 200])
# print('Predict (15):', add(10, 5))
# print()
# print('--------------------')


# @nn
# def first_char(a: String) -> String[1]:
#     a = a or ' '
#     return a[0]

# print()
# print('Result  ("abc"):', repr(first_char['abc']))
# print('Predict ("bca"):', repr(first_char('bca')))
# print()
# print('--------------------')


# @nn
# def concat(a: String[1], b: String[1]) -> String[2]:
#     a = a or ' '
#     b = b or ' '
#     return a[0] + b[0]

# print()
# print('Result  ("a!"):', repr(concat['a', '!']))
# print('Predict ("bT"):', repr(concat('b', 'T')))
# print()
# print('--------------------')



# @nn
# def normalize(tup: Tuple[Int, Int, Int]) -> Tuple[Float, Float, Float]:
#     x, y, z = tup
#     length = math.sqrt(x ** 2 + y ** 2 + z ** 2)
#     return x / length, y / length, z / length

# print()
# print('Result  (1, 0, 1):', normalize[[1, 0, 1]])
# print('Predict (1, 0, 1):', normalize((1, 0, 1)))
# print()
# print('--------------------')


@struct
class Car:
    name: String[10]
    num_wheels: Int

car = Car(name='Honda', num_wheels=4)

print(car)

print(car.name, car.num_wheels)
print(car[0], car[1])

print([i for i in car])
