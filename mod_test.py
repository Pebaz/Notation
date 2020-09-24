from random import randint
from keras.models import Sequential
from keras.layers import Dense

INPUT_NODES = 3
OUTPUT_NODES = 3

model = Sequential([
    Dense(OUTPUT_NODES, input_shape=(INPUT_NODES,))
])

model.compile(loss='mse', optimizer='adam')

input_data = [[randint(1, 100) for _ in range(3)] for i in range(1000000)]
output_data = [[i + 1 for i in data] for data in input_data]

print()

model.fit(input_data, output_data)

get_result = lambda prediction: list(prediction[0])
input_ = [10, 11, 12]
pred = model.predict([input_])

print()

print(get_result(pred))


