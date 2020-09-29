




from keras.models import Model
from keras.layers import Input, Dense


def create_model(inputs, outputs):
    inp = Input(shape=(inputs,))
    out = Dense(outputs)(inp)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='mse', optimizer='adam')
    return model

data = 'this is something that should be kept hidden'
passcode = 'who me'  # Expected password

as_layer = lambda s, size: [ord(c) for c in (s + ' ' * (size - len(s)))]

mod = create_model(len(passcode), len(data))
mod.fit(
    [
        as_layer(passcode, len(passcode))
    ] * 100000,
    [
        as_layer(data, len(data))
    ] * 100000
)


result = mod.predict([
    as_layer('hello', len(passcode))  # Wrong password
])[0]

string = [chr(max(0, min(0x110000 - 1, round(c)))) for c in result]

print(''.join(string))


result = mod.predict([
    as_layer('who me', len(passcode))  # Right password
])[0]

string = [chr(max(0, min(0x110000 - 1, round(c)))) for c in result]

print(''.join(string))
