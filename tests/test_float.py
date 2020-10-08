import random
import pytest
from nndt import *


def test_float_constructor():
    assert Float
    assert Float(3)
    assert Float(100)
    assert Float(100000)
    assert Float(2302309230)
    assert Float(-12323)
    assert Float(-4)
    assert Float(-0)
    assert Float(-11111111)

    for _ in range(250_000):
        assert Float(random.randint(-2147483648, 2147483647))


def test_float_to():
    assert Float(3).to() == 3
    assert Float(4).to() == 4
    assert Float(-3).to() == -3
    assert Float(-123).to() == -123
    assert Float(300).to() == 300
    assert Float(100000000000).to() == 100000000000
    assert Float(-100000000000).to() == -100000000000

    for _ in range(250_000):
        num = random.randint(-2147483648, 2147483647)
        assert Float(num).to() == num


def test_float_as_layer():
    assert Float(3).as_layer() == [3]
    assert Float(4).as_layer() == [4]
    assert Float(-3).as_layer() == [-3]
    assert Float(-123).as_layer() == [-123]
    assert Float(300).as_layer() == [300]
    assert Float(100000000000).as_layer() == [100000000000]
    assert Float(-100000000000).as_layer() == [-100000000000]

    for _ in range(250_000):
        num = random.randint(-2147483648, 2147483647)
        assert Float(num).as_layer() == [num]


def test_float_from_layer():
    assert Float.from_layer([3]).to() == 3
    assert Float.from_layer([4]).to() == 4
    assert Float.from_layer([-3]).to() == -3
    assert Float.from_layer([-123]).to() == -123
    assert Float.from_layer([300]).to() == 300
    assert Float.from_layer([100000000000]).to() == 100000000000
    assert Float.from_layer([-100000000000]).to() == -100000000000

    for _ in range(250_000):
        num = random.randint(-2147483648, 2147483647)
        assert Float.from_layer([num]).to() == num


def test_float_random():
    for _ in range(250_000):
        assert isinstance(Float.random().to(), float)


def test_float_type():
    assert str(Float) == '<Float[1]>'
    assert repr(Float(3)) == '<Float[1] 3.0>'

    with pytest.raises(TypeError):
        # TODO(pebaz): Won't let me step floato TypeError here in debugger. Why?
        assert f'{Float}' == '<Float[1]>'

    assert len(Float) == 1
    assert len(Float(3)) == 1


def test_float_nn():
    @nn
    def add(number_a: Float, number_b: Float) -> Float:
        return number_a + number_b

    assert add[100, 200] == 300
    result = add(1000, 2000)
    assert 2000 < result < 4000, 'Not even close'

