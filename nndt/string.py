import random
from . nndt import NNDT, _VariableLength


class String(_VariableLength, NNDT):
    """
    String class of shape 255.
    Can be customized to have any length using subscript: String[10]
    Value inputs shorter than SHAPE get padded with `\0`.
    """
    SHAPE = 5
    MIN_CODE = 0
    MAX_CODE = 0x110000
    VALID_UNICODE = lambda c: max(min(c, String.MAX_CODE), String.MIN_CODE)
    UNICODES = ''.join(
        chr(i)
        for i in range(MIN_CODE, 0x110000)
        if chr(i).isprintable()
    )

    def __init__(self, value):
        "Create new string, padding it with zeroes if shorter than SHAPE."
        assert len(value) <= self.SHAPE, f'String len capped at {self.SHAPE}'
        self.value = value + '\0' * (self.SHAPE - len(value))

    def __str__(self):
        display_val = self.value.replace('\0', '_')
        return f'<{self.__class__.__name__}[{self.SHAPE}] {repr(display_val)}>'

    def as_layer(self):
        return [ord(c) for c in self.value]

    @classmethod
    def from_layer(cls, layer):
        """
        Factory method to return a new String object from a given layer.

        If a layer's value lies outside of the valid unicode sequence, it is
        truncated to fit.

        # TODO(pebaz): Should this only be true of repr() or str()?
        Null characters '\0' are replaced with spaces for clarity.
        """
        converted_chars = ''.join(
            (chr(cls.VALID_UNICODE(round(c))) for c in layer)
        )
        return String[cls.SHAPE](converted_chars.rstrip('\0'))

    @classmethod
    def random(cls, length_choice=None):
        length = length_choice or random.randint(0, cls.SHAPE)
        if not length:
            random_str_gen = ''
        else:
            random_str_gen = (random.choice(cls.UNICODES) for _ in range(length))
        return String[cls.SHAPE](''.join(random_str_gen))
