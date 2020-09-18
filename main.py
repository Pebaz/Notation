import inspect
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
    for argument in sig.parameters:
        arg = sig.parameters[argument]
        assert arg != inspect._empty, 'Arguments must be typed'
        assert issubclass(arg.annotation, NNDT), 'Arguments must be NNDTs'

def nn(func):
    """
    Returns a function with an associated NN whose input layer consists of as
    many nodes as there are arguments (handled per type).

    For example: def _(Int, Int) would return a NN with an input layer that
    contains 2 input nodes.
    """
    validate_signature(func)

    __pycache__ = Path('__pycache__')
    __pycache__.mkdir(exist_ok=True)

    nn_file = __pycache__ / f'{func.__name__}'

    if nn_file.exists():
        func.__model__ = load_model(nn_file)
    else:
        num_input_nodes = NNDT.length_of(func.__annotations__.values())

        model = Sequential([Dense(1, input_shape=(num_input_nodes,))])
        model.compile(loss='mse', optimizer='adam')
        model.save(nn_file)
        model.fit(data_input, data_label)

        func.__model__ = model

    def closure(*args):
        nonlocal nn_file




        result_accurate = func(*args)
        args = [[i] for i in args]
        result_predict = func.__model__.predict(args)
        func.__model__.fit(*args, [result_accurate])




        func.__model__.save(nn_file)

        # if it matches, stop using the stored function!

        print(result_accurate, result_predict[0])

        return result_accurate

    return closure

class NNDT:
    "Neural Network-Aware Data Type"
    SHAPE = 1

    def __len__(self):
        return self.SHAPE

    @staticmethod
    def length_of(*args):
        return sum(len(i) for i in args)

class Int(NNDT):
    def __init__(self, value):
        self.value = value
        self.shape = 1

class String(NNDT):
    SHAPE = 255

    def __init__(self, value):
        self.value = value

@nn
def negate(number: Int):
    return -number

print(negate(123))
