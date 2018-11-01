# Recurrent Neural Network
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential

model = Sequential()

model.add(LSTM(10,input_shape=(100,1)))
model.add(Dense(10, activation='relu'))

model.add(Dense(1,activation='sigmoid'))

# visible = Input(shape=(100,1))
# hidden1 = LSTM(10)(visible)
# hidden2 = Dense(10, activation='relu')(hidden1)
# output = Dense(1, activation='sigmoid')(hidden2)
# model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='recurrent_neural_network.png')