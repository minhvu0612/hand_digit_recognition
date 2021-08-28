import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from tensorflow.keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def format_input(features, labels, num_classes):
    # features = tf.convert_to_tensor(features)
    features = tf.expand_dims(features, axis = -1)
    labels = keras.utils.to_categorical(labels, num_classes)
    return features, labels

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_classes = 10

print('Shape of the dataset before formatting')
print('\nTraining Data:', x_train.shape, y_train.shape, sep = ' ')
print('Test Data:', x_test.shape, y_test.shape, sep = ' ')

x_train, y_train = format_input(x_train, y_train, num_classes)
x_test, y_test = format_input(x_test, y_test, num_classes)

print('\nShape of the dataset after formatting')
print('\nTraining Data:', x_train.shape, y_train.shape, sep = ' ')
print('Test Data:', x_test.shape, y_test.shape, sep = ' ')

batch_size = 128

# Data augmentation for decreasing overfitting
train_datagen = ImageDataGenerator(rotation_range=0.5, height_shift_range=0.2, shear_range=0.2,
                                   zoom_range = 0.2, rescale = 1.0/255.0, dtype = 'float32')

valid_datagen = ImageDataGenerator(rescale=1./255., dtype = 'float32')
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle = True)
valid_generator = valid_datagen.flow(x_test, y_test, batch_size = batch_size)

def Models(inputs):
    x = Conv2D(64, kernel_size=(5, 5),activation='relu', padding = 'same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, kernel_size=(3, 3), activation = 'relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(512, (3, 3), activation='relu', padding = 'same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu', kernel_regularizer=keras.regularizers.L2())(x)
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation = 'softmax')(x)

    return output

input = Input(shape = (28, 28, 1))
output = Models(input)

model = Model(inputs = input, outputs = output)

model.summary()

epochs = 100
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])
hist = model.fit(train_generator, epochs = epochs, verbose = 1, validation_data = valid_generator)
print("The model has successfully trained")

score = model.evaluate(valid_generator, verbose=0)
print('Test loss:', score[0])
print('\nTest accuracy:', score[1] * 100, '%')
model.save('mnist.h5')