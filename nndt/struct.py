from . nndt import NNDT, NNDTException
from . tuple_ import Tuple

class Struct(Tuple):
    DISPLAY_NAME = 'Struct'
    FIELDS = {}
    
    def __init__(self, *args, **kwargs):
        if args:
            Tuple.__init__(self, args)
        elif kwargs:
            Tuple.__init__(self, tuple(kwargs.values()))
            self.__dict__.update(kwargs)
        else:
            raise NNDTException('Struct constructor takes at least 1 argument.')

    def __getitem__(self, key):

        # TODO(pebaz): Put these methods into Tuple so that you can
        # just build a decorator from a class and return a Tuple

        if isinstance(key, int):
            return self.value[key]
        else:
            return self.__dict__[key]

    def __str__(self):
        return f'<{self.DISPLAY_NAME}[{self.SHAPE}] {repr(self.to())}>'

def struct(class_):
    for nndt_type in class_.__annotations__.values():
        assert issubclass(nndt_type, NNDT), 'Struct fields must be NNDTs'

    # HHHHHHHHHM what to do with names?

    result = Struct[tuple(class_.__annotations__.values())]
    result.FIELDS = class_.__annotations__
    result.DISPLAY_NAME = class_.__name__
    return result
