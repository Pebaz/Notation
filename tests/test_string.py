import random
import pytest
from nndt import *

alpha = 'abcdefghijklmnopqrstuvwxyz'
alpha += alpha.upper()

def test_string_constructor():
    assert String
    assert String[3]
    assert String[30]
    assert String[31]
    assert String[32]
    assert String[33]
    assert String[34]
    assert String[35]
    assert String[36]
    assert String[37000]
    assert String[50]('Hello World!')
    assert String[50]('Hello World!1')
    assert String[60]('Hello World!2')
    assert String[60]('Hello World!3' * 1)
    assert String[62]('Hello World!4' * 2)
    assert String[255]('Hello World!5' * 3)
    assert String[61]('Hello World!6' * 4)
    assert String[255]('Hello World!7' * 5)
    
    for _ in range(250_000):
        assert String(
            ''.join([random.choice(alpha) for _ in range(len(String))])
        )


def test_string_to():
    assert String[255]('Hello World!').to() == 'Hello World!'
    assert String[255]('Hello World!1').to() == 'Hello World!1'
    assert String[255]('Hello World!2').to() == 'Hello World!2'
    assert String[255]('Hello World!3' * 1).to() == 'Hello World!3' * 1
    assert String[255]('Hello World!4' * 2).to() == 'Hello World!4' * 2
    assert String[255]('Hello World!5' * 3).to() == 'Hello World!5' * 3
    assert String[255]('Hello World!6' * 4).to() == 'Hello World!6' * 4
    assert String[255]('Hello World!7' * 5).to() == 'Hello World!7' * 5

    for _ in range(250_000):
        assert String(
            ''.join([random.choice(alpha) for _ in range(len(String))])
        ).to()


def test_string_as_layer():
    assert len(String[255]('Hello World!').as_layer()) == 255
    assert len(String[255]('Hello World!1').as_layer()) == 255
    assert len(String[255]('Hello World!2').as_layer()) == 255
    assert len(String[255]('Hello World!3' * 1).as_layer()) == 255
    assert len(String[255]('Hello World!4' * 2).as_layer()) == 255
    assert len(String[255]('Hello World!5' * 3).as_layer()) == 255
    assert len(String[255]('Hello World!6' * 4).as_layer()) == 255
    assert len(String[255]('Hello World!7' * 5).as_layer()) == 255

    assert String[1]('!').as_layer() == [ord('!')]
    assert String[1]('1').as_layer() == [ord('1')]
    assert String[1]('2').as_layer() == [ord('2')]
    assert String[1]('3').as_layer() == [ord('3')]
    assert String[1]('4').as_layer() == [ord('4')]
    assert String[1]('5').as_layer() == [ord('5')]
    assert String[1]('6').as_layer() == [ord('6')]
    assert String[1]('7').as_layer() == [ord('7')]
    assert String[1]('8').as_layer() == [ord('8')]
    assert String[1]('9').as_layer() == [ord('9')]
    assert String[1]('0').as_layer() == [ord('0')]


def test_string_from_layer():
    assert String[5]('abcde').as_layer() == [97, 98, 99, 100, 101]

    len_ = 5
    for size in range(len_ + 1, 5000):
        result = String[len_]('abcde').as_layer()
        assert len(result) == len_
        assert isinstance(result, list)


def test_string_random():
    for _ in range(250_000):
        result = String[3].random()
        assert isinstance(result.to(), str)
        assert len(result) == 3


def test_string_type():
    assert str(String) == '<String[5]>'
    assert repr(String[4]('3')) == "<String[4] '3___'>"

    with pytest.raises(TypeError):
        # TODO(pebaz): Won't let me step stringo TypeError here in debugger. Why?
        assert f'{String}' == '<String[255]>'

    assert len(String) == 5
    assert len(String.random()) == 5
    assert len(String[3].random()) == 3


def test_string_nn():
    @nn
    def first(word: String[2]) -> String[1]:
        if not word: return word
        return word[0]

    assert first['ab'] == 'a'
    result = first('ab')
    # assert result in alpha, 'Not even close'
    assert isinstance(result, str)
    assert len(result) == 1
