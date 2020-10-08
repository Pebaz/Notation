import random
import pytest
from nndt import *


def test_int_constructor():
    assert Int
    assert Int(3)
    assert Int(100)
    assert Int(100000)
    assert Int(2302309230)
    assert Int(-12323)
    assert Int(-4)
    assert Int(-0)
    assert Int(-11111111)

    for _ in range(250_000):
        assert Int(random.randint(-2147483648, 2147483647))


def test_int_to():
    assert Int(3).to() == 3
    assert Int(4).to() == 4
    assert Int(-3).to() == -3
    assert Int(-123).to() == -123
    assert Int(300).to() == 300
    assert Int(100000000000).to() == 100000000000
    assert Int(-100000000000).to() == -100000000000

    for _ in range(250_000):
        num = random.randint(-2147483648, 2147483647)
        assert Int(num).to() == num


def test_int_as_layer():
    assert Int(3).as_layer() == [3]
    assert Int(4).as_layer() == [4]
    assert Int(-3).as_layer() == [-3]
    assert Int(-123).as_layer() == [-123]
    assert Int(300).as_layer() == [300]
    assert Int(100000000000).as_layer() == [100000000000]
    assert Int(-100000000000).as_layer() == [-100000000000]

    for _ in range(250_000):
        num = random.randint(-2147483648, 2147483647)
        assert Int(num).as_layer() == [num]


def test_int_from_layer():
    assert Int.from_layer([3]).to() == 3
    assert Int.from_layer([4]).to() == 4
    assert Int.from_layer([-3]).to() == -3
    assert Int.from_layer([-123]).to() == -123
    assert Int.from_layer([300]).to() == 300
    assert Int.from_layer([100000000000]).to() == 100000000000
    assert Int.from_layer([-100000000000]).to() == -100000000000

    for _ in range(250_000):
        num = random.randint(-2147483648, 2147483647)
        assert Int.from_layer([num]).to() == num


def test_int_random():
    for _ in range(250_000):
        assert isinstance(Int.random().to(), int)


def test_int_type():
    assert str(Int) == '<Int[1]>'
    assert repr(Int(3)) == '<Int[1] 3>'

    with pytest.raises(TypeError):
        # TODO(pebaz): Won't let me step into TypeError here in debugger. Why?
        assert f'{Int}' == '<Int[1]>'

    assert len(Int) == 1
    assert len(Int(3)) == 1


def test_int_nn():
    @nn
    def add(number_a: Int, number_b: Int) -> Int:
        return number_a + number_b

    assert add[100, 200] == 300
    result = add(1000, 2000)
    assert 2000 < result < 4000, 'Not even close'

