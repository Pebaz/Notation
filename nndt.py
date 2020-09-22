class NNDTType(type):
    nndt_types = {}

    def __new__(cls, name, bases, dict_):
        print(1, cls)
        print(2, name)
        print(3, bases)
        print(4, dict_)
        
        cls.nndt_types[name] = (bases, dict_)

        cls.name = name
        cls.bases = bases
        cls.dict_ = dict_

        new_type = super().__new__(cls, name, bases, dict_)
        cls.nndt_types[new_type] = name
        return new_type

    def __getitem__(self, key):
        bases, dict_ = self.nndt_types[self.__name__]

        #result = super().__new__(self.__class__, self.name, self.bases, self.dict_)
        result = super().__new__(self.__class__, self.__name__, bases, dict_)
        result.SHAPE = key
        return result

    def __str__(self):
        return f'<{self.__name__}[{self.SHAPE}]>'

class NNDT(metaclass=NNDTType):
    SHAPE = 1

    # def __new__(cls, *args, **kwargs):
    #     return super().__new__(cls, *args, **kwargs)

    # def __init__(self, value):
    #     self.value = value

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'<{self.__class__.__name__}[{self.SHAPE}] {repr(self.value)}>'

class String(NNDT):
    pass

print(5, NNDT)
print(6, NNDT(3))
print(7, NNDT[5])
print(8, NNDT[3](100))
print(9, String[100])
print(10, String[100]('Foo'))
