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

        self.func = func
        self.result_predict = None

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
        if self.nn_file.exists():
            self.__model__ = load_model(self.nn_file)
        else:
            self.__pycache__.mkdir(exist_ok=True)

            self.arg_types = [
                val for key, val in self.func.__annotations__.items()
                if key != 'return'
            ]

            num_input_nodes = NNDT.length_of(self.arg_types)

            self.__model__ = Sequential()
            self.__model__.add(Dense(1, input_shape=(num_input_nodes,)))
            self.__model__.compile(loss='mse', optimizer='adam')

            self.train()

    def train(self, enthusiasm=1000):
        """
        Automatically trains the NN using random inputs coupled with the correct
        return value obtained from the function.
        """
        # import numpy as np
        # data_input = np.random.normal(size=4)  # 1000000
        # data_label = -(data_input)
        # model.fit(data_input, data_label)

        data_input = []
        for _ in range(enthusiasm):
            layer = []
            for nndt_type in self.arg_types:
                layer.extend(nndt_type.random().as_layer())
            data_input.append(layer)



        data_label = [
            [self.call_raw(*i)] for i in data_input
        ]

    def layer_as_signature(self, nodes):
        signature = []
        ptr = 0
        for nndt_type in self.arg_types:
            node_slice = nodes[ptr:ptr + nndt_type.SHAPE]
            instance = nndt_type.from_layer(node_slice)
            signature.append(instance)
            ptr += nndt_type.SHAPE

    def __cleanup__(self):
        "Save the NN to file!"
        #self.__model__.save(self.nn_file)

    def __call__(self, *args):
        "Call the function and the NN, returning the most accurate value."
        result_accurate = self.func(*args)
        args = [[i] for i in args]
        self.result_predict = self.__return_type__(
            self.__model__.predict(args)
        )
        self.__model__.fit(*args, [result_accurate])

        # if it matches, stop using the stored function!

        #print(result_accurate, result_predict)

        return result_accurate

    def call_raw(self, *args, **kwargs):
        "Bypass NN entirely."
        return self.func(*args, **kwargs)

    def call_predicted(self, *args):
        "Ensure use of NN's prediction."
        self(*args)
        return self.result_predict
        

class NNDT:
    "Neural Network-Aware Data Type"
    SHAPE = 1

    def __init__(self, value):
        self.value = value

    def __len__(self):
        return self.SHAPE

    def to(self):
        raise NotImplemented()

    @staticmethod
    def length_of(*args):
        return sum(len(i) for i in args)

    def __str__(self):
        return str(self.value)

    def as_layer(self):
        raise NotImplemented()


class Int(NNDT):
    def to(value):
        return int(value)

    @staticmethod
    def random():
        return Int(random.randint(-2147483648, 2147483647))

    def as_layer(self):
        return [self.value]


class String(NNDT):
    SHAPE = 255

    def __init__(self, value):
        assert len(value) < self.SHAPE, f'String len capped at {self.SHAPE}'
    
    def to(self):
        'Return a string'

    @classmethod
    def random(cls):
        length = random.randint(0, cls.SHAPE)
        return String()


@NNFunc
def negate(number: Int) -> Int:
    return -number


print()
print('Result  :', negate(123))
print('Predict :', negate.call_predicted(123))
print()
print(Int.random())
print('--------------------')
