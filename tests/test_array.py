import random
import pytest
from nndt import *


def test_array_constructor():
    assert Array
    assert Array[1, Int]
    assert Array[1, Float]
    assert Array[1, String[1]]
    assert Array[1, Array[1, Int]]
    assert Array[1, Array[1, Float]]
    assert Array[1, Array[1, String[1]]]
    assert Array[1, Array[1, Array[1, Int]]]
    assert Array[1, Array[1, Array[1, Float]]]
    assert Array[1, Array[1, Array[1, String[1]]]]

    with pytest.raises(AssertionError):
        assert Array([1])

    assert Array[1, Int]([1])
    assert Array[1, Float]([1.0])
    assert Array[1, String[1]](['!'])
    assert Array[1, Array[1, Int]]([[1]])
    assert Array[1, Array[1, Float]]([[1.0]])
    assert Array[1, Array[1, String[1]]]([['!']])
    assert Array[1, Array[1, Array[1, Int]]]([[[1]]])
    assert Array[1, Array[1, Array[1, Float]]]([[[1.0]]])
    assert Array[1, Array[1, Array[1, String[1]]]]([[['!']]])


def test_array_to():
    with pytest.raises(AssertionError):
        assert Array([1]).to()

    assert Array[1, Int]([1]).to() == [1]
    assert Array[1, Float]([1.0]).to() == [1.0]
    assert Array[1, String[1]](['!']).to() == ['!']
    assert Array[1, Array[1, Int]]([[1]]).to() == [[1]]
    assert Array[1, Array[1, Float]]([[1.0]]).to() == [[1.0]]
    assert Array[1, Array[1, String[1]]]([['!']]).to() == [['!']]
    assert Array[1, Array[1, Array[1, Int]]]([[[1]]]).to() == [[[1]]]
    assert Array[1, Array[1, Array[1, Float]]]([[[1.0]]]).to() == [[[1.0]]]
    assert Array[1, Array[1, Array[1, String[1]]]]([[['!']]]).to() == [[['!']]]

    assert Array[3, String[3]](['ABC', '10', '!']).to() == ['ABC', '10', '!']


def test_array_as_layer():
    with pytest.raises(AssertionError):
        assert Array([1]).as_layer()

    assert Array[1, Int]([1]).as_layer() == [1]
    assert Array[1, Float]([1.0]).as_layer() == [1.0]
    assert Array[1, String[1]](['!']).as_layer() == [33]
    assert Array[1, Array[1, Int]]([[1]]).as_layer() == [1]
    assert Array[1, Array[1, Float]]([[1.0]]).as_layer() == [1.0]
    assert Array[1, Array[1, String[1]]]([['!']]).as_layer() == [33]
    assert Array[1, Array[1, Array[1, Int]]]([[[1]]]).as_layer() == [1]
    assert Array[1, Array[1, Array[1, Float]]]([[[1.0]]]).as_layer() == [1.0]
    assert Array[1, Array[1, Array[1, String[1]]]]([[['!']]]).as_layer() == [33]

    assert Array[3, String[3]](['ABC', '10', '!']).as_layer()


def test_array_from_layer():
    assert Array[1, Int].from_layer(Array[1, Int]([1]).as_layer()).to() == [1]
    assert Array[1, Float].from_layer(
        Array[1, Float]([1.0]).as_layer()
    ).to() == [1.0]
    assert Array[1, String[1]].from_layer(
        Array[1, String[1]](['!']).as_layer()
    ).to() == ['!']
    assert Array[1, Array[1, Int]].from_layer(
        Array[1, Array[1, Int]]([[1]]).as_layer()
    ).to() == [[1]]
    assert Array[1, Array[1, Float]].from_layer(
        Array[1, Array[1, Float]]([[1.0]]).as_layer()
    ).to() == [[1.0]]
    assert Array[1, Array[1, String[1]]].from_layer(
        Array[1, Array[1, String[1]]]([['!']]).as_layer()
    ).to() == [['!']]
    assert Array[1, Array[1, Array[1, Int]]].from_layer(
        Array[1, Array[1, Array[1, Int]]]([[[1]]]).as_layer()
    ).to() == [[[1]]]
    assert Array[1, Array[1, Array[1, Float]]].from_layer(
        Array[1, Array[1, Array[1, Float]]]([[[1.0]]]).as_layer()
    ).to() == [[[1.0]]]
    assert Array[1, Array[1, Array[1, String[1]]]].from_layer(
        Array[1, Array[1, Array[1, String[1]]]]([[['!']]]
    ).as_layer()).to() == [[['!']]]

    assert Array[3, String[3]](['ABC', '10', '!']).as_layer()
    assert Array[3, String[3]].from_layer(Array[3, String[3]](['ABC', '10', '!']).as_layer())
    assert Array[3, String[3]].from_layer(Array[3, String[3]](['ABC', '10', '!']).as_layer()).to()


def test_array_random():
    assert isinstance(Array[3, Int].random().to(), list)
    assert isinstance(Array[3, Float].random().to(), list)
    assert isinstance(Array[3, String[3]].random().to(), list)
    assert isinstance(Array[100, Int].random().to(), list)
    assert isinstance(Array[random.randint(1, 1000), Int].random().to(), list)
    assert isinstance(Array[random.randint(1, 1000), Float].random().to(), list)
    assert isinstance(
        Array[random.randint(1, 1000), String[1]].random().to(), list
    )
    assert isinstance(
        Array[
            random.randint(1, 1000), random.choice([Int, Float, String[1]])
        ].random().to(), list
    )



def test_array_type():
    assert str(Array) == '<Array[1, None]>'
    assert str(Array[1, Int]) == '<Array[1, Int[1]]>'
    assert str(Array[1, Float]) == '<Array[1, Float[1]]>'
    assert str(Array[1, String[1]]) == '<Array[1, String[1]]>'
    assert str(Array[1, Array[1, Int]]) == '<Array[1, Array[1, Int[1]]]>'

    assert repr(Array) == '<Array[1, None]>'
    assert repr(Array[1, Int]) == '<Array[1, Int[1]]>'
    assert repr(Array[1, Float]) == '<Array[1, Float[1]]>'
    assert repr(Array[1, String[1]]) == '<Array[1, String[1]]>'
    assert repr(Array[1, Array[1, Int]]) == '<Array[1, Array[1, Int[1]]]>'

    with pytest.raises(TypeError):
        # TODO(pebaz): Won't let me step into TypeError here in debugger. Why?
        assert f'{Array}' == '<Array[1, None]>'
        assert f'{Array[1, Int]}' == '<Array[1, Int[1]]>'
        assert f'{Array[1, Float]}' == '<Array[1, Float[1]]>'
        assert f'{Array[1, String[1]]}' == '<Array[1, String[1]]>'
        assert f'{Array[1, Array[1, Int]]}' == '<Array[1, Array[1, Int[1]]]>'

    assert len(Array) == 1
    assert len(Array[1, Int]) == 1
    assert len(Array[1, Float]) == 1
    assert len(Array[1, String[1]]) == 1
    assert len(Array[1, Array[1, Int]]) == 1

def test_array_nn():
    @nn
    def append(array: Array[1, Int], num: Int) -> Array[2, Int]:
        return array + [num]

    assert append[[1], 2] == [1, 2]
    result = append([1], 2)
    assert 1 in result or 2 in result, 'Prediction was *way* off'

