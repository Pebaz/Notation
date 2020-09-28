from . tuple_ import Tuple

class Struct(Tuple):
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):

        # TODO(pebaz): Put these methods into Tuple so that you can
        # just build a decorator from a class and return a Tuple

        if isinstance(key, int):
            return self.value[key]
        else:
            return self.__dict__[key]


def struct(class_):
    for nndt_type in class_.__annotations__.values():
        assert isinstance(nndt_type, NNDT), 'FOOOOOOOOOOOO'

    # HHHHHHHHHM what to do with names?

    #return Struct[*class_.__annotations__.values()]
