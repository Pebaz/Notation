# Notation

A proof of concept that compiles each function to Neural Networks and trains
them automatically between invocations, substituting whichever value is more
accurate.

### Features

  * Create automatically-trained neural network in 4 lines of code
  * Supports calling bundled Python function for 100% accurate results
  * Use subscript syntax to make calling bundled Python function convenient
  * Define custom Neural Network-Aware Data Types as needed
  * Basic NN-Data Types are included:
    * Bool
    * Int
    * Float
    * String
    * Array
  * Collection types can be customized by indexing them
  * Stores the generated model in the `__pycache__` directory along with the
    hash of the function source. If the function source is updated, the model is
    regenerated. If not, the trained model is loaded from disk saving time.

### Example

```python
from nndt import *

@nn
def lor(bitTrue: Bool, bit2: Bool) -> Bool:
    "Logical OR Function using neural network"
    return bitTrue or bit2

# Call Auto-Trained Neural Network
assert lor(True, False) == True
assert lor(False, False) == False
assert lor(True, True) == True
assert lor(False, True) == True

# Call Actual Python Function
assert lor[True, False] == True
assert lor[False, False] == False
assert lor[True, True] == True
assert lor[False, True] == True


@nn
def land(bitTrue: Bool, bit2: Bool) -> Bool:
    "Logical AND Function using neural network"
    return bitTrue and bit2

assert land(True, False) == False
assert land(False, False) == False
assert land(True, True) == True
assert land(False, True) == False

assert land[True, False] == False
assert land[False, False] == False
assert land[True, True] == True
assert land[False, True] == False


# If desired, generate additional random training inputs and validated outputs,
# and run through more cycles of training:
lor.train(1000)
land.train(1000)

```


### Customizable Built In Types

```python
array_of_3_ints = Array[3, Int]([1, 2, 3])
array_of_2_arrays_of_3_bools = Array[2, Array[3, Bool]]([[0, 1, 0], [1, 1, 1]])

string_of_length_3 = String[3]('HI!')
string_of_length_5 = String('Hello')  # Str len defaults to 5
```


## How It Works

Notation is based off of the fact that [neural networks are able to compute any function](http://neuralnetworksanddeeplearning.com/chap4.html).

For instance, for a given function `int add(int a, int b)`, there exists a
neural network that is able to compute it's results.

Notation provides a decorator `@nn` that is able to generate, train, validate,
and predict values using a neural network that is attached to the function.

It accomplishes this by returning a custom class (rather than the function) from
the `@nn` decorator. This custom class `NNFunc` acts like a normal function in
that it can be `__call__()`ed. The result of calling the "function" is a
prediction from the attached neural network. To get a result from the original
Python function, you can use subscript notation to pass arguments rather than
positional arguments. An example is shown below:

```python
@nn
def add(a: Int, b: Int) -> Int:
    return a + b

print(add(1, 2))  # 3 is predicted
print(add[1, 2])  # 3 is obtained by calling underlying Python function
```

The body of the Python function is used to automatically train the attached
model or provide the means to get a 100% accurate result.

In order to support the calculation of any function, a set of neural network
input and output nodes needs to be created for each and every function that 100%
matches the given arguments to the function. This is a tedious but doable
process.

Notation accomplishes this by proving a set of data types that represent a
Python data type that can be marshalled to and from an input/output layer.

This can be visualized below:

Once data types can be passed to and returned from a `NNFunc`, there needs to be
a way to tell the attached neural network about them.

Notation uses Python type annotations to accomplish this. There are some rules
that `NNFunc`s need to adhere to when using the `@nn` decorator:

  1. All arguments must have a data type annotation
  2. All type annnotations must inherit from the NNDT data type
  3. The function must have a return type annotation
  4. The return type annotation must inherit from the NNDT data type
  5. Function must be pure (have no side-effects)

When wrapping a given Python function annotated with these types, Notation uses
the `SHAPE` of each one to determine the total size of the input layer
(arguments) and the output layer (return type).

Each of the builtin NNDTs have a preset `SHAPE` that corresponds to the number
of nodes needed to store it's representation within a neural network layer.

For instance, the `Int` NNDT has a preset `SHAPE` of `1` because it only takes a
single node to store it within a layer.

This poses a subtle problem, however, because by using this system, no
variable-length data types can be simulated as inputs/outputs to a given
function which severely limits the usefulness of Notation.

Through the use of metaclasses, Notation provides a base NNDTType type that has
the capability to customize the `SHAPE` of a given type by *generating a new one
whenever needed*. This can be visualized in the following code example:

```python
@nn
def capitalize(word: String) -> String:
    "String has to have a set SHAPE of say, 255"
    return word.upper()
```

As it currently stands, the String data type can only support a constant
`SHAPE`. If the `SHAPE` is 255, a shorter input would result in lots of
zero-padding. The way to fix this is to create a new String class with a new
`SHAPE` for each function that expects a String argument:

```python
class PhoneNumber(String):
    SHAPE = 10
    ...

@nn
def call(number: PhoneNumber) -> PhoneNumber:
    "The input string can have custom SHAPE"
```

However, this will *certainly* get tedious for more than the trivialest of
cases. This is exactly why all NNDTs have a metaclass `NNDTType` that overrides
subscript notation to support `SHAPE` customization ***per argument!***

```python
# No need for PhoneNumber class since SHAPE can be customized directly on String

@nn
def call(number: String[10]) -> String[10]:
    "Fully-customizable SHAPE per-argument, per-return type, per-function!"
```

This system is the most flexible and allows on-the-fly creation of classes with
a given shape. Customizable NNDTs in Notation include `String` and `Array`:

```python
@nn
def foo(a1: String[3], a2: Array[3, Int], a3: Array[3, String[10]]) -> Int:
    "NN input and output layer shapes can be determined statically!"
```

Once the input and output shape of a given function/model is determined, how
does training work?

So far, the following items can be determined statically:

  * Accurate function results (using function body)
  * Input/output layer shape (using data type annotations)

If these items are combined with the ability to randomly generate valid inputs,
*the model could be trained with them automatically*.

To support this, Neural Network Aware Data Types must provide a `.random()`
function that generates a new instance of that type with a random value. This
function is used to train the neural network automatically by generating random
inputs, passing them to the underlying Python function to get an accurate
result, and then using the random inputs along with the accurate outputs to
train the model.

By using all of these facilities, Notation is able to convert a given Python
function into a neural network model seamlessly at module load time.
