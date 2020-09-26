import random
from . nndt import NNDT, _Container

#arr = Array[10, String[3]]  # 10 * 3 = _Container


class Array(_Container, NNDT):
    def __init__(self, value):
        size = len(value)
        assert size <= self.COUNT, f'Input array too large: {size}/{self.COUNT}'

        self.value = [self.OF_TYPE(element) for element in value]

        complete_size = NNDT.length_of(*self.value)
        assert complete_size <= self.SIZE, (
            f'Input elements marshalled into a size greater than the array can '
            f'store: {complete_size}/{self.SHAPE}'
        )

    def as_layer(self):
        layer = []
        for element in self.value:
            layer.extend(element.as_layer())

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
        assert size == self.SHAPE, f'Input layer too small: {size}/{self.SHAPE}'

        # elements = []
        # ptr = 0
        # for _ in range(self.COUNT):
        #     node_slice = layer[ptr:ptr + len(self.OF_TYPE)]
        #     instance = self.OF_TYPE.from_layer(node_slice)
        #     elements.append(instance)
        #     ptr += len(self.OF_TYPE)

        elements = []
        for ptr in range(0, self.COUNT - 1, len(self.OF_TYPE)):
            node_slice = layer[ptr:ptr + len(self.OF_TYPE)]
            instance = self.OF_TYPE.from_layer(node_slice)
            elements.append(instance)

        return self.__class__(elements)

    @classmethod
    def random(cls):
        elements = [self.OF_TYPE.random() for _ in range(self.COUNT)]
        return self.__class__(elements)
