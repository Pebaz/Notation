import random
import pytest
from nndt import Array, String, Int, Float


def test_array_constructor():
    # index[]
    # different types using []
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
    assert Array[3, Int].random()
    assert Array[3, Float].random()
    assert Array[3, String[3]].random()
    assert Array[100, Int].random()
    assert Array[random.randint(1, 1000), Int].random()
    assert Array[random.randint(1, 1000), Float].random()
    assert Array[random.randint(1, 1000), String[1]].random()
    assert Array[
        random.randint(1, 1000), random.choice([Int, Float, String[1]])
    ].random()



def test_array_type():
    # str()
    # repr()
    # format()
    # len()
    pass


def test_array_nn():
    pass
