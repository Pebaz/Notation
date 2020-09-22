"""
"""

class _VariableLength:
    "Mixin class to ensure that staticly sized classes don't get indexed."

    def __len__(self):
        "Length is equal to shape plus the shape of all contained values."
        return self.SHAPE + sum(len(v) for v in self.value)


class NNDTException(Exception):
    "Base Exception for NNDT library."


class NNDTIndexException(NNDTException):
    def __init__(self, class_, key):
        self.class_ = class_
        self.key = key

    def __str__(self):
        return (
            f'{self.class_.__name__} has a static SHAPE of {self.class_.SHAPE}, '
            f'cannot be set to {self.key}'
        )


class NNDTType(type):
    """
    Metaclass for a range of Neural Network-aware Data Types.

    Supports variable shaped NN input layers by allowing indexing the class
    itself. For instance, an Int has a shape of 1 since it is a single node that
    gets passed to the NN. However, an Array needs to be able to support any
    number of nodes based on usage. To accomplish this, you can index the Array
    class directly to create a brand new type that has shape of N.
    """
    nndt_types = {}

    def __new__(cls, name, bases, dict_):        
        cls.nndt_types[name] = bases, dict_
        return super().__new__(cls, name, bases, dict_)

    def __getitem__(self, key):
        "Create unique class (not object) with custom `SHAPE`."

        bases, dict_ = self.nndt_types[self.__name__]

        if _VariableLength not in bases:
            raise NNDTIndexException(self, key)

        dict_['SHAPE'] = key

        return super().__new__(
            self.__class__,
            self.__name__,
            bases,
            dict_
        )

    def __str__(self):
        return f'<{self.__name__}[{self.SHAPE}]>'


class NNDT(metaclass=NNDTType):
    "Neural Network-Aware Data Type"
    SHAPE = 1

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'<{self.__class__.__name__}[{self.SHAPE}] {repr(self.value)}>'

    def __len__(self):
        return self.SHAPE

    def to(self):
        return self.value

    @staticmethod
    def length_of(*args):
        return sum(len(i) for i in args)

    def as_layer(self):
        pass

    @staticmethod
    def from_layer(layer):
        pass


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


class String(NNDT, _VariableLength):
    """
    String class of shape 255.
    Can be customized to have any length using subscript: String[10]
    Value inputs shorter than SHAPE get padded with `\0`.
    """
    SHAPE = 255
    UNICODES = ''.join(
        chr(i)
        for i in range(32, 0x110000)
        if chr(i).isprintable()
    )

    def __init__(self, value):
        "Create new string, padding it with zeroes if shorter than SHAPE."
        assert len(value) <= self.SHAPE, f'String len capped at {self.SHAPE}'
        self.value = value + '\0' * (self.SHAPE - len(value))

    def as_layer(self):
        return [ord(c) for c in self.value]

    def from_layer(layer):
        return String(''.join(chr(c) for c in layer))

    @classmethod
    def random(cls, length_choice=None):
        length = length_choice or random.randint(0, cls.SHAPE)
        random_str_gen = (random.choice(cls.UNICODES) for _ in range(length))
        return String(''.join(random_str_gen))
