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


class NNPointer:
    def __init__(self, func):
        atexit.register(self.shutdown)

    def shutdown(self):
        "Save the NN to file!"
        

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

@nn
def negate(number: Int) -> Int:
    return -number

print(negate(123))
