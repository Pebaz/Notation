"""
Samuel Wilder 9/29/2020

Obfuscates a given piece of data (in this case a string) by hiding within an ML
model.

To get that data back out, a string matching the password must be passed in.

All other input returns garbled text.
"""

DATA = 'this is something that should be kept hidden'
PASSCODE = 'who me'  # Expected password

from keras.models import Model
from keras.layers import Input, Dense

def create_model(inputs, outputs):
    "Creates a simple model with 1 input & output layer with the given sizes."
    inp = Input(shape=(inputs,))
    out = Dense(outputs)(inp)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='mse', optimizer='adam')
    return model

# Turns a string into a list of ints matching length of input/output layer
as_layer = lambda s, size: [ord(c) for c in (s + ' ' * (size - len(s)))]

mod = create_model(len(PASSCODE), len(DATA))

num_iterations = 100000

mod.fit(
    [
        as_layer(PASSCODE, len(PASSCODE))
    ] * num_iterations,
    [
        as_layer(DATA, len(DATA))
    ] * num_iterations
)

# ------------------------------------------------------------------------------
result = mod.predict([
    as_layer('hello', len(PASSCODE))  # Wrong password
])[0]

string = [chr(max(0, min(0x110000 - 1, round(c)))) for c in result]

print(''.join(string))


# ------------------------------------------------------------------------------
result = mod.predict([
    as_layer('who me', len(PASSCODE))  # Right password
])[0]

string = [chr(max(0, min(0x110000 - 1, round(c)))) for c in result]

print(''.join(string))

# mod.save('compress')
