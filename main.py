from nndt import *




@nn
def negate(number: Int) -> Int:
    return -number


print()
print('Result  (-123):', negate(123))
print('Predict (-124):', negate.call_predicted(124))
print()
print('--------------------')
# # negate.train()


# @nn
# def add(a: Int, b: Int) -> Int:
#     return a + b

# print()
# print('Result  (300):', add(100, 200))
# print('Predict (15):', add.call_predicted(10, 5))
# print()
# print('--------------------')


# @nn
# def first_char(a: String) -> String:
#     return a[0]

# print()
# print('Result  ("abc"):', first_char("abc"))
# print('Predict ("bca"):', first_char.call_predicted("bca"))
# print()
# print('--------------------')

#import ipdb; ipdb.set_trace()
