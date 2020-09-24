from keras.models import Sequential
from keras.layers import Dense

model = Sequential([Dense(3, input_shape=(3,))])

model.compile(loss='mse', optimizer='adam')

input_data = [[i for i in range(3)], [i for i in range(3)]]
output_data = [[i + 1 for i in data] for data in input_data]

print()

model.fit(input_data, output_data)

print([*model.predict([[10, 11, 12]])[0]])
