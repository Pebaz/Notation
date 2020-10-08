import math
from nndt import *


def test_nn_arrays():

    @nn
    def append(array: Array[1, Int], num: Int) -> Array[2, Int]:
        return array + [num]

    assert append[[1], 2] == [1, 2]
    result = append([1], 2)
    assert 1 in result or 2 in result, 'Prediction was *way* off'


def test_nn_ints():

    almost_equal = lambda x, y: x == y or x == y + 1 or x == y - 1

    @nn
    def negate(number: Int) -> Int:
        return -number
 
    assert negate[123] == -123
    a, b = negate(124), -124
    assert almost_equal(a, b), f'Not even close: {a} != {b}'

    @nn
    def add(a: Int, b: Int) -> Int:
        return a + b
    

    assert add[100, 200] == 300
    a, b = add(8000, 11), 8011
    assert 7900 < a < 8100, f'Not even close: {a} != {b}'


def test_nn_strings():

    @nn
    def first_char(a: String) -> String[1]:
        a = a or ' '
        return a[0]

    assert first_char['abc'] == 'a'
    assert first_char('bca')

    @nn
    def concat(a: String[1], b: String[1]) -> String[2]:
        a = a or ' '
        b = b or ' '
        return a[0] + b[0]

    assert concat['a', '!'] == 'a!'
    assert concat('b', 'T')
