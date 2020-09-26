import random
from . nndt import NNDT, _Container


# TODO(pebaz): Switch meaning of SHAPE and len()



class Array(_Container, NNDT):
    def __init__(self, value):
        size = len(value)
        assert size <= self.COUNT, f'Input array too large: {size}/{self.COUNT}'

        self.value = []
        for element in value:
            if isinstance(element, NNDT):
                self.value.append(element)  # Don't marshall, already done
            else:  # It's a Python object, marshall it
                self.value.append(self.OF_TYPE(element))

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

        # elements = []
        # ptr = 0
        # for _ in range(cls.COUNT):
        #     node_slice = layer[ptr:ptr + len(cls.OF_TYPE)]
        #     instance = cls.OF_TYPE.from_layer(node_slice)
        #     elements.append(instance)
        #     ptr += len(cls.OF_TYPE)

        elements = []
        for ptr in range(0, cls.COUNT - 1, len(cls.OF_TYPE)):
            node_slice = layer[ptr:ptr + len(cls.OF_TYPE)]
            instance = cls.OF_TYPE.from_layer(node_slice)
            elements.append(instance)





















        THIS IS NOT RIGHT:
        Array[3, String[3]].from_layer(Array[3, String[3]].random().as_layer())
































        return cls(elements)

    @classmethod
    def random(cls):
        return cls([cls.OF_TYPE.random() for _ in range(cls.COUNT)])