"""
"""

import inspect
import atexit
from pathlib import Path
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense


__all__ = 'nn', 'NNDT', 'NNDTException', 'NNDTIndexException'


class _CustomizableType:
    ""

    @staticmethod
    def customize(dict_, key):
        "Override as you see fit."


class _VariableLength(_CustomizableType):
    "Mixin class to ensure that staticly sized classes don't get indexed."

    @staticmethod
    def customize(dict_, key):
        dict_['SHAPE'] = key

    # def __len__(self):
    #     "Length is equal to shape plus the shape of all contained values."
    #     return self.SHAPE + sum(len(v) for v in self.value)



class _Container(_CustomizableType):
    """
    Mixin class that allows collection types like Array.

    Makes this possible: Array[10, String[3]]
    Total Shape: 30  # 10 * 3

    Type: Array[2, Array[10, String[3]]]
    Total Shape: 60  # 2 * 10 * 3
    """

    @staticmethod
    def customize(dict_, key):
        size, of_type = key
        dict_['SHAPE'] = size * len(of_type)
        dict_['OF_TYPE'] = of_type
        dict_['COUNT'] = size

    # TODO(pebaz): Should len(Array[3, Array[10, Int]]) == 30? Or 3/len() == 30?
    # def __len__(self):
    #     return self.SHAPE * len(self.OF_TYPE)


class _Struct(_CustomizableType):
    """
    Mixin class 

    Struct[Int, Int, Float, String[10]]
    """

    @staticmethod
    def customize(dict_, nndt_types):
        dict_['SHAPE'] = NNDT.length_of(*nndt_types)
        dict_['TYPES'] = nndt_types


class NNDTException(Exception):
    "Base Exception for NNDT library."


class NNDTIndexException(NNDTException):
    def __init__(self, class_, key):
        self.class_ = class_
        self.key = key

    def __str__(self):
        return (
            f'{self.class_.__name__} has a static SHAPE of {self.class_.SHAPE}, '
            f'cannot be set to {self.key}'
        )


class NNDTType(type):
    """
    Metaclass for a range of Neural Network-aware Data Types.

    Supports variable shaped NN input layers by allowing indexing the class
    itself. For instance, an Int has a shape of 1 since it is a single node that
    gets passed to the NN. However, an Array needs to be able to support any
    number of nodes based on usage. To accomplish this, you can index the Array
    class directly to create a brand new type that has shape of N.
    """
    nndt_types = {}

    def __new__(cls, name, bases, dict_):        
        cls.nndt_types[name] = bases, dict_
        return super().__new__(cls, name, bases, dict_)

    def __getitem__(self, key):
        "Create unique class (not object) with custom `SHAPE`."

        bases, dict_ = self.nndt_types[self.__name__]

        customizer = bases[0]

        if not issubclass(customizer, _CustomizableType):
            raise Exception(
                'Type cannot be customized. '
                'Is a subclass of _CustomizableType first in inheritance?'
            )

        # Perform custom construction per customizer type
        customizer.customize(dict_, key)

        return super().__new__(
            self.__class__,
            self.__name__,
            bases,
            dict_
        )

    def __str__(self):
        return f'<{self.__name__}[{self.SHAPE}]>'

    def __repr__(self):
        return str(self)

    def __format__(self):
        return str(self)

    def __len__(self):
        return self.SHAPE


class NNDT(metaclass=NNDTType):
    "Neural Network-Aware Data Type"
    SHAPE = 1

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return f'<{self.__class__.__name__}[{self.SHAPE}] {repr(self.value)}>'

    def __repr__(self):
        return str(self)

    def __format__(self):
        return str(self)

    def __len__(self):
        return self.SHAPE

    def to(self):
        return self.value

    @staticmethod
    def length_of(*args):
        return sum(len(i) for i in args)

    def as_layer(self):
        pass

    @staticmethod
    def from_layer(layer):
        pass

    def as_pyobj(self):
        "Returns the given NNDT as a Python-compatible object."
        return self.value


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

    def create_model(self, num_inputs, num_outputs):
        inputs = Input(shape=(num_inputs,))
        outputs = Dense(num_outputs)(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer='adam')
        return model

    def initialize_function(self):
        self.arg_types = [
            val for key, val in self.func.__annotations__.items()
            if key != 'return'
        ]

        if self.nn_file.exists():
            self.__model__ = load_model(self.nn_file)
        else:
            self.__pycache__.mkdir(exist_ok=True)
            num_input_nodes = NNDT.length_of(*self.arg_types)
            num_output_nodes = len(self.__return_type__)
            self.__model__ = self.create_model(
                num_input_nodes, num_output_nodes
            )
            self.train()

    def train(self, enthusiasm=10000):
        """
        Automatically trains the NN using random inputs coupled with the correct
        return value obtained from the function.
        """

        python_inputs = []
        data_inputs = []
        for _ in range(enthusiasm):
            signature_layers = [
                nndt_type.random() for nndt_type in self.arg_types
            ]
            python_inputs.append([layer.value for layer in signature_layers])

            # Flatten the layer into one big layer and append it as an input
            data_inputs.append([
                node for layer in signature_layers for node in layer.as_layer()
            ])

        data_outputs = []
        for training_layer in python_inputs:
            python_output = self.call_raw(*training_layer)
            output_as_layer = self.__return_type__(python_output).as_layer()
            data_outputs.append(output_as_layer)

        self.__model__.fit(data_inputs, data_outputs)

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





'''


class Struct(NNDT, _VariableLength):
    "Car = Struct[Int, Float, String[10], Array[3], Struct[Int, Int, Int]]"

# DATA CLASSES https://docs.python.org/3/library/dataclasses.html

class NNDTStruct:
    """
    Go and find each field that doesn't start with __ and then pack it's type
    into a struct automatically. Marshalling and unmarshalling the fields work
    as expected for each inner type.
    """

class Color(NNDTStruct):
    Red: Float
    Green: Float
    Blue: Float

class MyUserType(NNDTStruct):
    NumWheels = Int
    NumDoors = Int
    Name = String[15]
    CarColor = Color

    def uppercase_name(self):
        return self.Name.upper()

    # TODO(pebaz): Support NN methods!
    # @nn
    # def uppercase_name(self) -> String[15]:
    #     ...


# TODO(pebaz): Could have entire type system with operator overloading...
# TODO(pebaz): Need to make a Struct class that introspects class fields to find
# TODO(pebaz):     shape of object.


# Array[String[10]]  (array of 1 string of length 10)
# Array[3, String[10]]  (array of 3 strings of length 10)

# TODO(pebaz): Support Array[3, String[10]] for array of phone numbers
# TODO(pebaz): This needs to have the right SHAPE and len.
# TODO(pebaz): Shouldn't the __getitem__ method support key.SHAPE? or len(key)?
'''



