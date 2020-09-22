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
    SHAPE = 1

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'<{self.__class__.__name__}[{self.SHAPE}] {repr(self.value)}>'

class String(NNDT, _VariableLength):
    pass

print(5, NNDT)
print(6, NNDT(3))
print(7, NNDT[5])
print(8, NNDT[3](100))
print(9, String[100])
print(10, String[100]('Foo'))


# Array[String[10]]  (array of 1 string of length 10)
# Array[3, String[10]]  (array of 3 strings of length 10)

# TODO(pebaz): Support Array[3, String[10]] for array of phone numbers
# TODO(pebaz): This needs to have the right SHAPE and len.

