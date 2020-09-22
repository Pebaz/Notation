class NNDTType(type):

    def __new__(cls, name, bases, dict_):
        print(1, cls)
        print(2, name)
        print(3, bases)
        print(4, dict_)
        cls.name = name
        cls.bases = bases
        cls.dict_ = dict_
        return super().__new__(cls, name, bases, dict_)

    def __getitem__(self, key):
        # generic_type = NNDTType.__new__(self, self.name, self.bases, self.dict_)
        # generic_type.SHAPE = key
        # return generic_type
        self.SHAPE = key
        return self

    def __str__(self):
        return f'<{self.__class__.__name__}[{self.SHAPE}]>'

class NNDT(type, metaclass=NNDTType):
    SHAPE = 1

    # def __new__(cls, *args, **kwargs):
    #     return super().__new__(cls, *args, **kwargs)

    # def __init__(self, value):
    #     self.value = value

    def __str__(self):
        return f'<{self.__class__.__name__}[{self.SHAPE}] {self.value}>'

print(5, NNDT)
print(6, NNDT[5])
# print(7, NNDT(3).SHAPE)
# print(8, NNDT[int].SHAPE)
