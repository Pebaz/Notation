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
