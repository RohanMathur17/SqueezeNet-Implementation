import model
import os
import cv2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import ReLU
from sklearn.model_selection import train_test_split

datasets = tf.contrib.keras.datasets
layers =  tf.contrib.keras.layers
models = tf.contrib.keras.models
losses = tf.contrib.keras.losses
optimizers = tf.contrib.keras.optimizers 
metrics = tf.contrib.keras.metrics
preprocessing_image = tf.contrib.keras.preprocessing.image
utils = tf.contrib.keras.utils
callbacks = tf.contrib.keras.callbacks


X = []
y = []

# Input your own data here
base_dir = '/content/gdrive/MyDrive/SqueezeNet/data/'

print('[SPLITTING DATASET...]')
for f in sorted(os.listdir(base_dir)):

    if os.path.isdir(base_dir+f):
       
        for i in sorted(os.listdir(base_dir+f)):
            X.append(base_dir+f+'/'+i)
            y.append(f)

print('[SPLIT!]')


(trainX, testX, trainY, testY) = train_test_split(X,y,
	test_size=0.25,random_state=42)

print('[RESIZING IMAGES...]')

trainX_list = []
for imagePath in trainX:
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    trainX_list.append(image)

testX_list = []
for imagePath in testX:
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    testX_list.append(image)

trainY_array = np.unique(trainY, return_inverse=True)[1]
trainY_array = trainY_array.reshape(len(trainY_array),1)

testY_array = np.unique(testY, return_inverse=True)[1]
testY_array = testY_array.reshape(len(testY_array),1)

trainX_list_array = np.asarray(trainX_list)
testX_list_array  = np.asarray(testX_list)

print('[DONE!]')
print('[PREPROCESSING IMAGES...]')

train_datagen = preprocessing_image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

test_datagen = preprocessing_image.ImageDataGenerator(rescale=1./255)


trainY_array = utils.to_categorical(trainY_array, num_classes=5)
testY_array = utils.to_categorical(testY_array, num_classes=5)

train_generator = train_datagen.flow(x=trainX_list_array, y=trainY_array, batch_size=32, shuffle=True)

test_generator = test_datagen.flow(x=testX_list_array, y=testY_array, batch_size=32, shuffle=True)


def compile_model(model):

    # loss
    loss = losses.categorical_crossentropy

    # optimizer
    optimizer = optimizers.RMSprop(lr=0.0001)

    # metrics
    metric = [metrics.categorical_accuracy, metrics.top_k_categorical_accuracy]

    # compile model with loss, optimizer, and evaluation metrics
    model.compile(optimizer, loss, metric)

    return model


input_shape = (224,224,3)
classes = 5 # Change accordingly
sn = model.SqueezeNet(input_shape = input_shape,
					  nclasses = classes)

print('[MODEL BUILT...]')
sn = compile_model(sn)

print('[TRAINING MODEL...]')

history = sn.fit_generator(
    train_generator,
    steps_per_epoch=400,
    epochs=200,
    validation_data=test_generator,
    validation_steps=200)

def plot_accuracy_and_loss(history):
    plt.figure(1, figsize= (15, 10))

    # plot train and test accuracy
    plt.subplot(221)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('SqueezeNet accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # plot train and test loss
    plt.subplot(222)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('SqueezeNet loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.show()
    
plot_accuracy_and_loss(history)