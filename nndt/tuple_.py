import random
from . nndt import NNDT, _Struct


class Tuple(_Struct, NNDT):
    """
    The only difference between a Tuple and a Struct is that a tuple can be
    customized every time it is used (i.e. Tuple[Int, Int], Tuple[Int, Float])
    while a struct is a class that has Python functions that manipulate that
    specific set of data and so the fields are constant between each usage
    (i.e. Car <- there is no customizer, Car == Tuple[Int, String[10]] + Python
    functions that manipulate the data).
    """
    def __init__(self, value):
        size = len(value)
        assert size <= self.SHAPE, f'Input array too large: {size}/{self.SHAPE}'

        self.value = []
        for i, (element, nndt_type) in enumerate(zip(value, self.TYPES)):
            # It's a Python object, marshall it
            if not isinstance(element, NNDT):
                element = nndt_type(element)

            # NOTE(pebaz): No subclass support since isinstance won't =(
            assert element.__class__ == nndt_type, (
                f'Expected element at index {i} to be of type {nndt_type}, got '
                f'{element.__class__} instead'
            )
            self.value.append(element)

        self.value = tuple(self.value)

        complete_size = NNDT.length_of(*self.value)
        assert complete_size <= self.SHAPE, (
            f'Input elements marshalled into a size greater than the array can '
            f'store: {complete_size}/{self.SHAPE}'
        )

    def as_layer(self):
        layer = []
        for element in self.value:
            layer.extend(element.as_layer())
        return layer

    @classmethod
    def from_layer(cls, layer):
        """"""
        size = len(layer)
        assert size == len(cls), f'Input layer too small: {size}/{len(cls)}'

        elements = []
        ptr = 0
        for nndt_type in cls.TYPES:
            chunk_size = len(nndt_type)
            node_slice = layer[ptr:ptr + chunk_size]
            instance = nndt_type.from_layer(node_slice)
            elements.append(instance)
            ptr += chunk_size

        return cls(elements)

    @classmethod
    def random(cls):
        return cls([nndt_type.random() for nndt_type in cls.TYPES])

    def to(self):
        return tuple(element.to() for element in self.value)

'''
@struct  # Returns Tuple[Int, String[10]]
class Car:
    NumWheels: Int
    Make: String[10]
'''
