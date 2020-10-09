import random
import pytest
from nndt import *
import random
import pytest
from nndt import *


def test_bool_constructor():
    assert Bool('a')
    assert Bool('')
    assert Bool([1])
    assert Bool([])
    assert Bool(tuple())
    assert Bool((1,))
    assert Bool({})
    assert Bool({'a': 1})
    assert Bool(set())
    assert Bool(set([1]))
    assert Bool(frozenset())
    assert Bool(frozenset([1]))

    for _ in range(250_000):
        assert Bool(random.randint(-2147483648, 2147483647))


def test_bool_to():
    bools = 0, 1
    for _ in range(250_000):
        assert Bool.random().to() in bools


def test_bool_as_layer():
    bools = 0, 1
    for _ in range(250_000):
        assert Bool.random().as_layer()[0] in bools


def test_bool_from_layer():
    for _ in range(250_000):
        assert isinstance(Bool.from_layer(random.choice([[1], [0]])).to(), bool)


def test_bool_random():
    for _ in range(250_000):
        assert isinstance(Bool.random().to(), bool)


def test_bool_type():
    assert str(Bool) == '<Bool[1]>'
    assert repr(Bool(3)) == '<Bool[1] True>'

    with pytest.raises(TypeError):
        # TODO(pebaz): Won't let me step into TypeError here in debugger. Why?
        assert f'{Bool}' == '<Bool[1]>'

    assert len(Bool) == 1
    assert len(Bool(3)) == 1


def test_bool_nn():
    @nn
    def lor(bit1: Bool, bit2: Bool) -> Bool:
        return bit1 or bit2

    assert lor(1, 0) == True
    assert lor(0, 0) == False
    assert lor(1, 1) == True
    assert lor(0, 1) == True


    @nn
    def land(bit1: Bool, bit2: Bool) -> Bool:
        return bit1 and bit2

    assert land(1, 0) == False
    assert land(0, 0) == False
    assert land(1, 1) == True
    assert land(0, 1) == False