import inspect
import atexit
import random
from pathlib import Path
from keras.models import Sequential, load_model
from keras.layers import Dense


class NNFunc:
    """
    Couples a function with an associated NN whose input layer consists of as
    many nodes as there are arguments (handled per type).

    For example: def _(Int, Int) would return a NN with an input layer that
    contains 2 input nodes.
    """
    def __init__(self, func):
        self.validate_signature(func)
        self.use_prediction_only = False

        self.func = func

        self.__return_type__ = func.__annotations__['return']

        self.__pycache__ = Path('__pycache__')

        self.nn_file = self.__pycache__ / f'{func.__name__}'

        self.initialize_function()

        atexit.register(self.__cleanup__)

    def validate_signature(self, func):
        sig = inspect.signature(func)
        assert issubclass(sig.return_annotation, NNDT), (
            'Must have return type of NNDT'
        )
        for argument in sig.parameters:
            arg = sig.parameters[argument]
            assert arg != inspect._empty, 'Arguments must be typed'
            assert issubclass(arg.annotation, NNDT), 'Arguments must be NNDTs'

    def initialize_function(self):
        self.arg_types = [
            val for key, val in self.func.__annotations__.items()
            if key != 'return'
        ]

        if self.nn_file.exists():
            self.__model__ = load_model(self.nn_file)
        else:
            self.__pycache__.mkdir(exist_ok=True)
            num_input_nodes = NNDT.length_of(self.arg_types)

            self.__model__ = Sequential()
            self.__model__.add(Dense(1, input_shape=(num_input_nodes,)))
            self.__model__.compile(loss='mse', optimizer='adam')

            self.train()

    def train(self, enthusiasm=5):
        """
        Automatically trains the NN using random inputs coupled with the correct
        return value obtained from the function.
        """

        data_input = []
        for _ in range(enthusiasm):
            layer = []
            for nndt_type in self.arg_types:
                layer.extend(nndt_type.random().as_layer())
            data_input.append(layer)

        data_label = []
        for training_layer in data_input:
            signature = self.layer_as_signature(training_layer)
            marshalled_values = [nndt_inst.value for nndt_inst in signature]
            data_label.append(self.call_raw(*marshalled_values))

        self.__model__.fit(data_input, data_label)

    def layer_as_signature(self, nodes):
        signature = []
        ptr = 0
        
        for nndt_type in self.arg_types:
            node_slice = nodes[ptr:ptr + nndt_type.SHAPE]
            instance = nndt_type.from_layer(node_slice)
            signature.append(instance)
            ptr += nndt_type.SHAPE

        return signature

    def __cleanup__(self):
        "Save the NN to file!"
        self.__model__.save(self.nn_file)

    def signature_as_layer(self, args):
        marshalled_values = []
        for arg, nndt_type in zip(args, self.arg_types):
            marshalled_values.extend(nndt_type(arg).as_layer())
        return [marshalled_values]

    def __call__(self, *args):
        "Call the function and the NN, returning the most accurate value."

        result_accurate = self.func(*args)

        input_values = self.signature_as_layer(args)
        prediction = self.__model__.predict(input_values)
        nndt_prediction = self.__return_type__.from_layer(prediction[0])
        result_predict = nndt_prediction.to()
        nndt_return_value = self.__return_type__(result_accurate)

        self.__model__.fit(input_values, [nndt_return_value.as_layer()])

        # If it matches, stop using the stored function!
        if result_accurate == result_predict:
            self.certify()
        elif self.use_prediction_only:
            return result_predict

        return result_accurate

    def certify(self):
        # TODO(pebaz): Load this value from a text file called: func.cerfity
        self.use_prediction_only = True

    def call_raw(self, *args, **kwargs):
        "Bypass NN entirely."
        return self.func(*args, **kwargs)

    def call_predicted(self, *args):
        "Ensure use of NN's prediction."
        tmp = self.use_prediction_only
        self.use_prediction_only = True
        result = self(*args)
        self.use_prediction_only = tmp
        return result


nn = NNFunc


class NNDT:
    "Neural Network-Aware Data Type"
    SHAPE = 1

    def __init__(self, value):
        self.value = value

    def __len__(self):
        return self.SHAPE

    def to(self):
        return self.value

    @staticmethod
    def length_of(*args):
        return sum(len(i) for i in args)

    def __str__(self):
        return str(self.value)

    def as_layer(self):
        raise NotImplemented()

    @staticmethod
    def from_layer(layer):
        raise NotImplemented()


class Int(NNDT):
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


class String(NNDT):
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


# @nn
# def negate(number: Int) -> Int:
#     return -number


# print()
# print('Result  (-123):', negate(123))
# print('Predict (-124):', negate.call_predicted(124))
# print()
# print('--------------------')
# # negate.train()



# @nn
# def add(a: Int, b: Int) -> Int:
#     return a + b

# print()
# print('Result  (300):', add(100, 200))
# print('Predict (15):', add.call_predicted(10, 5))
# print()
# print('--------------------')


@nn
def first_char(a: String) -> String:
    return a[0]

print()
print('Result  ("abc"):', first_char("abc"))
print('Predict ("bca"):', first_char.call_predicted("bca"))
print()
print('--------------------')

#import ipdb; ipdb.set_trace()
