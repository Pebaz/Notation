import random
from . nndt import *

class Bool(NNDT):
    def __init__(self, value):
        value = bool(value)
        assert value in (True, False), 'Value must be truthy or falsey'
        NNDT.__init__(self, value)

    def as_layer(self):
        return [int(self.value)]

    @classmethod
    def random(cls):
        return cls(random.choice((True, False)))

    @classmethod
    def from_layer(cls, layer):
        return cls(round(layer[0]))


TRUE = Bool(True)
FALSE = Bool(False)
