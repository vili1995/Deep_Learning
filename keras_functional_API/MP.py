# Multilayer Perceptron
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.models import Sequential


#model
model = Sequential()

model.add(Dense(10, input_shape=(10,), activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='sigmoid'))


# visible = Input(shape=(10,))
# hidden1 = Dense(10, activation='relu')(visible)
# hidden2 = Dense(20, activation='relu')(hidden1)
# hidden3 = Dense(10, activation='relu')(hidden2)
# output = Dense(1, activation='sigmoid')(hidden3)
# model = Model(inputs=visible, outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model, to_file='multilayer_perceptron_graph.png')