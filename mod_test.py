from random import randint
from keras.models import Model, Sequential
from keras.layers import Input, Dense

INPUT_NODES = 2
OUTPUT_NODES = 3

# get rid of sequential, use functional api

def get_model(num_inputs, num_outputs):
    inputs = Input(shape=(num_inputs,))
    outputs = Dense(num_outputs)(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer='adam')
    return model


model = get_model(INPUT_NODES, OUTPUT_NODES)
input_data = [
    [randint(1, 100) for _ in range(INPUT_NODES)]
    for i in range(1000000)
]
output_data = [[i + 1 for i in data] + [data[-1] + 2] for data in input_data]
print(input_data[0], output_data[0])
model.fit(input_data, output_data)
get_result = lambda prediction: list(prediction[0])
input_ = [10, 11]
pred = model.predict([input_])
print()
print(get_result(pred))
