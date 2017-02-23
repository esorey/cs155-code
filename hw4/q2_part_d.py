import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.datasets import mnist
import os

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# One-hot encode the labels
y_train = keras.utils.np_utils.to_categorical(y_train)
y_test = keras.utils.np_utils.to_categorical(y_test)

# Reshape the input images to be 1d vectors instead of 2d vectors
X_train = X_train.reshape(len(X_train), 28**2)
X_test = X_test.reshape(len(X_test), 28**2)

# Rescale the pixel values to be between 0 and 1
X_train, X_test = X_train / 255, X_test / 255


##### Question D: Modeling Part 2

# Build the model
model = Sequential()
model.add(Dense(150, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.3))


model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

fit = model.fit(X_train, y_train, batch_size=100, nb_epoch=30,
   verbose=1)

#Evaluate the model on the test set
score = model.evaluate(X_test, y_test, verbose=0)
print('\nTest score:', score[0])
print('Test accuracy:', score[1])

# Make a sound so I check the results
os.system('say "done"')
