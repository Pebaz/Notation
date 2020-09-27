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
        for element, nndt_type in zip(value, self.TYPES):
            if isinstance(element, NNDT):
                self.value.append(element)  # Don't marshall, already done
            else:  # It's a Python object, marshall it
                self.value.append(nndt_type(element))
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
        """
        Factory method to return a new String object from a given layer.

        If a layer's value lies outside of the valid unicode sequence, it is
        truncated to fit.

        # TODO(pebaz): Should this only be true of repr() or str()?
        Null characters '\0' are replaced with spaces for clarity.
        """
        size = len(layer)
        assert size == cls.SHAPE, f'Input layer too small: {size}/{cls.SHAPE}'

        chunk_size = len(cls.OF_TYPE)
        elements = []
        for ptr in range(0, cls.COUNT * chunk_size, chunk_size):
            node_slice = layer[ptr:ptr + chunk_size]
            instance = cls.OF_TYPE.from_layer(node_slice)
            elements.append(instance)

        return cls(elements)

    @classmethod
    def random(cls):
        return cls([cls.OF_TYPE.random() for _ in range(cls.COUNT)])

    def to(self):
        return [element.to() for element in self.value]

'''
@struct  # Returns Tuple[Int, String[10]]
class Car:
    NumWheels: Int
    Make: String[10]
'''
