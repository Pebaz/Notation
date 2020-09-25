import random
from . nndt import NNDT


class Int(NNDT):
    """
    Integer class of shape 1.
    Supports numbers from -2147483648 to 2147483647.
    """
    def to(self):
        "Return an int and round out as much innacuracy as possible."
        return int(self.value)

    @staticmethod
    def random():
        return Int(random.randint(-2147483648, 2147483647))

    def as_layer(self):
        return [self.value]

    @staticmethod
    def from_layer(layer):
        (layer_value,) = layer
        return Int(layer_value)


class Float(NNDT):
    """
    Float class of shape 1.
    Supports numbers from
    -340282346638528859811704183484516925440.0 to
    340282346638528859811704183484516925440.0.
    """
    def to(self):
        "Return an int and round out as much innacuracy as possible."
        return float(self.value)

    @staticmethod
    def random():
        return Float(random.uniform(
            -340282346638528859811704183484516925440.0,
            340282346638528859811704183484516925440.0
        ))

    def as_layer(self):
        return [self.value]

    @staticmethod
    def from_layer(layer):
        (layer_value,) = layer
        return Float(layer_value)


