# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 21:59:54 2019

@author: Brian Chan
"""

from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)     
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')     
X_test = X_test.astype('float32')     
X_train /= 255     
X_test /= 255


classes = 10
Y_train = np_utils.to_categorical(Y_train, classes)     
Y_test = np_utils.to_categorical(Y_test, classes)

input_size = 784
batch_size = 100     
hidden_neurons = 400     
epochs = 30

model = Sequential()     
model.add(Dense(hidden_neurons, input_dim=input_size)) 
model.add(Activation('relu'))     
model.add(Dense(classes, input_dim=hidden_neurons)) 
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adadelta')

model.summary()

history = model.fit(X_train, Y_train, validation_split = 0.5, batch_size=batch_size, epochs=epochs, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1]) 

from matplotlib import pyplot as plt
#history = model
#history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()