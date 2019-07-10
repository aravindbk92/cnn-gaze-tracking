#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 00:00:13 2018

@author: abk
"""
import cv2
import numpy as np
import keras
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

from keras.layers import BatchNormalization
from keras.constraints import maxnorm
from keras.optimizers import Adam

from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint

from keras.utils import np_utils
from keras import backend as K

X = np.load("image_data.npy")
y = np.load("mouse_data.npy")
face_data = np.load("face_data.npy")

train_data = X.astype('float32')/255.0

train_labels = y
num_classes = 2

#cv_data = train_data[200:,:]
#cv_labels = train_labels[200:,:]
#train_data = train_data[:200,:]
#train_labels = train_labels[:200,:]

#cv_data = cv_data.reshape(-1, 15, 30, 1)
train_data = train_data.reshape(-1, 15, 30, 1)

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(15, 30, 1), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.26))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(BatchNormalization(axis=1))

model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.26))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.26))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.26))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(BatchNormalization(axis=1))

model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.26))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.26))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.26))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=1))
model.add(BatchNormalization(axis=1))

model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.26))
model.add(Flatten())
model.add(Dense(1308, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.432))
model.add(Dense(num_classes))

# Compile model
epochs = 100

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# checkpoint
filepath="weightsbest.h5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

# Fit the model
model.fit(train_data, train_labels, epochs=epochs, batch_size=20, callbacks=[checkpoint])
# Final evaluation of the model
#scores = model.evaluate(cv_data, cv_labels, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('100epochs.h5')