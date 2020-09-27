from nndt import *

# @nn
# def append(array: Array[1, Int], num: Int) -> Array[2, Int]:
#     return array + [num]


# print()
# print('Result  ([1, 2]):', append([1], 2))
# print('Predict ([1, 2]):', append.call_predicted([1], 2))
# print()
# print('--------------------')


# @nn
# def negate(number: Int) -> Int:
#     return -number


# print()
# print('Result  (-123):', negate(123))
# print('Predict (-124):', negate.call_predicted(124))
# print()
# print('--------------------')


@nn
def add(a: Int, b: Int) -> Int:
    return a + b

print()
print('Result  (300):', add(100, 200))
print('Predict (15):', add.call_predicted(10, 5))
print()
print('--------------------')


# @nn
# def first_char(a: String) -> String[1]:
#     return a[0]

# print()
# print('Result  ("abc"):', repr(first_char('abc')))
# print('Predict ("bca"):', repr(first_char.call_predicted('bca')))
# print()
# print('--------------------')


# @nn
# def concat(a: String[1], b: String[1]) -> String[2]:
#     return a[0] + b[0]

# print()
# print('Result  ("a!"):', repr(concat('a', '!')))
# print('Predict ("bT"):', repr(concat.call_predicted('b', 'T')))
# print()
# print('--------------------')
