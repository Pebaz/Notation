import inspect
import atexit
from pathlib import Path
from keras.models import Sequential, load_model
from keras.layers import Dense

# if Path('FirstModel.ml').exists():
#     model = load_model('FirstModel')
# else:
#     model = Sequential([
#         Dense(1, input_shape=(1,))
#     ])

#     model.compile(loss='mse', optimizer='adam')

#     import numpy as np
#     data_input = np.random.normal(size=1000000)
#     data_label = -(data_input)

#     model.fit(data_input, data_label)
#     model.save('FirstModel')

# print(model.predict([5]))

def validate_signature(func):
    sig = inspect.signature(func)
    assert issubclass(sig.return_annotation, NNDT), (
        'Must have return type of NNDT'
    )
    for argument in sig.parameters:
        arg = sig.parameters[argument]
        assert arg != inspect._empty, 'Arguments must be typed'
        assert issubclass(arg.annotation, NNDT), 'Arguments must be NNDTs'


class NNFunc:
    def __init__(self, func):
        validate_signature(func)

        self.func = func
        self.result_predict = None

        self.__return_type__ = func.__annotations__['return']

        self.__pycache__ = Path('__pycache__')

        self.nn_file = self.__pycache__ / f'{func.__name__}'

        self.initialize_function()

        atexit.register(self.__cleanup__)

    def initialize_function(self):
        if self.nn_file.exists():
            self.__model__ = load_model(self.nn_file)
        else:
            self.__pycache__.mkdir(exist_ok=True)

            arg_types = [
                val for key, val in func.__annotations__.items()
                if key != 'return'
            ]

            num_input_nodes = NNDT.length_of(arg_types)

            model = Sequential([Dense(1, input_shape=(num_input_nodes,))])
            model.compile(loss='mse', optimizer='adam')

            # import numpy as np
            # data_input = np.random.normal(size=4)  # 1000000
            # data_label = -(data_input)

            model.fit(data_input, data_label)

            self.__model__ = model

    def __cleanup__(self):
        "Save the NN to file!"
        self.__model__.save(self.nn_file)

    def __call__(self, *args):
        "Call the function and the NN, returning the most accurate value."
        result_accurate = self.func(*args)
        args = [[i] for i in args]
        self.result_predict = self.__return_type__.from_(
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
        

def nn(func):
    """
    Returns a function with an associated NN whose input layer consists of as
    many nodes as there are arguments (handled per type).

    For example: def _(Int, Int) would return a NN with an input layer that
    contains 2 input nodes.
    """
    validate_signature(func)

    func.__return_type__ = func.__annotations__['return']

    __pycache__ = Path('__pycache__')
    __pycache__.mkdir(exist_ok=True)

    nn_file = __pycache__ / f'{func.__name__}'

    if nn_file.exists():
        func.__model__ = load_model(nn_file)
    else:
        arg_types = [
            val for key, val in func.__annotations__.items()
            if key != 'return'
        ]
        print('->', func.__annotations__)
        num_input_nodes = NNDT.length_of(arg_types)

        model = Sequential([Dense(1, input_shape=(num_input_nodes,))])
        model.compile(loss='mse', optimizer='adam')

        import numpy as np
        data_input = np.random.normal(size=4)  # 1000000
        data_label = -(data_input)

        model.fit(data_input, data_label)

        model.save(nn_file)
        func.__model__ = model

    def closure(*args):
        result_accurate = func(*args)
        args = [[i] for i in args]
        result_predict = func.__return_type__.from_(
            func.__model__.predict(args)
        )
        func.__model__.fit(*args, [result_accurate])

        # if it matches, stop using the stored function!

        print(result_accurate, result_predict)

        return result_accurate

    return closure

class NNDT:
    "Neural Network-Aware Data Type"
    SHAPE = 1

    def __init__(self, value):
        self.value = value

    def __len__(self):
        return self.SHAPE

    @staticmethod
    def from_(value):
        return None

    @staticmethod
    def length_of(*args):
        return sum(len(i) for i in args)

class Int(NNDT):
    @staticmethod
    def from_(value):
        return int(value)

class String(NNDT):
    SHAPE = 255
    def __init__(self, value):
        assert len(value) < self.SHAPE, f'String len capped at {self.SHAPE}'

@NNFunc
def negate(number: Int) -> Int:
    return -number

print()
print('Result:', negate(123))
print()
