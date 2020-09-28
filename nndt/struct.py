from . nndt import NNDT, NNDTException
from . tuple_ import Tuple

class Struct(Tuple):
    """
    @struct
    class Car:
        name: String[10]
        num_wheels: Int

    car = Car(name='Honda', num_wheels=4)

    print(car)

    print(car.name, car.num_wheels)
    print(car[0], car[1])

    print([i for i in car])
    """
    DISPLAY_NAME = 'Struct'
    FIELDS = {}
    
    def __init__(self, *args, **kwargs):
        if args:
            Tuple.__init__(self, args)
            self.__dict__.update({
                key : val for key, val in zip(self.FIELDS.keys(), args)
            })
        elif kwargs:
            Tuple.__init__(self, tuple(kwargs.values()))
            self.__dict__.update(kwargs)
        else:
            raise NNDTException('Struct constructor takes at least 1 argument.')

    def __getitem__(self, key):
        return self.__dict__[tuple(self.FIELDS.keys())[key]]

    def __str__(self):
        return f'<{self.DISPLAY_NAME}[{self.SHAPE}] {repr(self.to())}>'


def struct(class_):
    """
    Constructs new class with metadata to support dot operator attribute access.
    """
    for nndt_type in class_.__annotations__.values():
        assert issubclass(nndt_type, NNDT), 'Struct fields must be NNDTs'

    result = Struct[tuple(class_.__annotations__.values())]
    result.FIELDS = class_.__annotations__
    result.DISPLAY_NAME = class_.__name__
    return result
