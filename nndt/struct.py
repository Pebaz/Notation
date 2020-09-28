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
