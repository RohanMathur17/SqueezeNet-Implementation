## Model Architecture of SqueezeNet

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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, ReLU
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import AvgPool2D
from tensorflow.keras.layers import concatenate

# Fire Module
def fire_module(x,s1,e1,e3):

    #x --> layer
    #s1 --> squeeze dimension
    #e1 --> expanding for 1x1 layer
    #e3 --> expanding for 3x3 layers

    # Squeezing Layer
    s1x = Conv2D(s1,kernel_size = 1, padding='same')(x)
    s1x = ReLU()(s1x)

    #1x1 expand Layer
    e1x = Conv2D(e1,kernel_size = 1, padding='same')(s1x)

    #3x3 expand Layer
    e3x = Conv2D(e3,kernel_size = 3, padding='same')(s1x)

    #Combining and Passing through ReLU Layer
    x = concatenate([e1x,e3x])
    x = ReLU()(x)

    return x

'''
SqueezeNet begins with a standalone convolution layer (conv1), 
followed by 8 Fire modules (fire1–8), ending with a final conv layer (conv10).
The number of filters per fire module is gradually increased from the beginning to the end of the network.
Max-pooling with a stride of 2 is performed after layers conv1, fire4, fire8, and conv10.
Dropout with ratio of 50% is applied after the fire9 module.

'''
# SqueezeNet Layer
def SqueezeNet(input_shape, nclasses):

    input = Input(input_shape)

    # 1st Convolution
    x = Conv2D(96,kernel_size = (7,7), 
                strides = (2,2),padding='same',input_shape = input_shape)(input)

    # 1st MaxPooling
    x = MaxPool2D((3,3),strides = (2,2),padding='same')(x)

    # FireModule1
    x = fire_module(x,s1 =16,e1=64,e3 = 64)

    # FireModule2
    x = fire_module(x,s1 =16,e1=64,e3 = 64)

    # FireModule3
    x = fire_module(x,s1=32,e1=128,e3=128)

    # 2nd MaxPooling
    x = MaxPool2D((3,3),strides = (2,2),padding='same')(x)

    # FireModule4
    x = fire_module(x,s1 = 32,e1=128,e3=128)

    # FireModule5
    x = fire_module(x,s1 =48,e1=192,e3 =192)

    # FireModule6
    x = fire_module(x,s1 =48,e1=192,e3 =192)

    # FireModule7
    x = fire_module(x,s1 =64,e1=256,e3 =256)
        
    # 3rd MaxPooling
    x = MaxPool2D((3,3),strides = (2,2),padding='same')(x)

    # FireModule8
    x = fire_module(x,s1=64,e1=256,e3=256)

    # 2nd Convolution
    x = Dropout(0.5)(x)

    # For Classes
    x = layers.Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)

    x = layers.Activation('relu', name='relu_conv10')(x)

    x = layers.GlobalAveragePooling2D()(x)

    out = layers.Activation('softmax', name='loss')(x)

    model = models.Model(input, out, name='squeezenet')
    
    return model