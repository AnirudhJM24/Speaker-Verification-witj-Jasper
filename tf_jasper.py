

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import keras
from keras.utils.vis_utils import plot_model




def JasperBlock(x,num_filters,kernel_size,dilation_rate,dropout_rate,repeats,train):
  residual = x

  for i in range(0,repeats-1):
    x = tf.keras.layers.Conv1D(filters=num_filters,kernel_size= kernel_size, padding='same', dilation_rate=dilation_rate,trainable=train)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

  x = tf.keras.layers.Conv1D(filters=num_filters,kernel_size= kernel_size, padding='same', dilation_rate=dilation_rate,trainable = train)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  residual = tf.keras.layers.Conv1D(filters=num_filters,kernel_size= kernel_size, padding='same')(residual)
  residual = tf.keras.layers.BatchNormalization()(residual)
  x = tf.keras.layers.add([x,residual],trainable=train)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Dropout(dropout_rate)(x)

  return x

def CreateSR(y, num_classes):
  y = tf.keras.layers.Flatten()(y)
  y = tf.keras.layers.Dense(256, activation = "relu")(y)
  y = tf.keras.layers.Dense(128, activation = "relu")(y)
  y = tf.keras.layers.Dense(num_classes, activation = "softmax")(y)
  # y = tf.keras.layers.Activation(num_classes)(y)


  return y

def JasperBlock2ret(x,num_filters,kernel_size,dilation_rate,dropout_rate, repeats):
  residual = x

  for i in range(0,repeats-1):
    x = tf.keras.layers.Conv1D(filters=num_filters,kernel_size= kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

  x = tf.keras.layers.Conv1D(filters=num_filters,kernel_size= kernel_size, padding='same', dilation_rate=dilation_rate)(x)
  x = tf.keras.layers.BatchNormalization()(x)
  residual = tf.keras.layers.Conv1D(filters=num_filters,kernel_size= kernel_size, padding='same')(residual)
  residual = tf.keras.layers.BatchNormalization()(residual)
  x = tf.keras.layers.add([x,residual])
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Dropout(dropout_rate)(x)

  return x, x

def create_Jasper(inputs,num_blocks,num_sublocks):
  fil = [256,384,512,640,768]
  ker = [11,13,17,21,25]

  x = tf.keras.layers.Conv1D(filters=256,kernel_size=11,strides=2,padding='same',dilation_rate=1)(inputs)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Dropout(0.2)(x)


  for i in range(num_blocks):
    if i==1:
      x,y = JasperBlock2ret(x,num_filters=fil[i],kernel_size=ker[i],dilation_rate=1,dropout_rate=0.2,repeats=3)
      y = CreateSR(y, 251)
    if i<1:
      x = JasperBlock(x,num_filters=fil[i],kernel_size=ker[i],dilation_rate=1,dropout_rate=0.2,repeats = 3,train = True)
    elif i>1:
      x = JasperBlock(x,num_filters=fil[i],kernel_size=ker[i],dilation_rate=1,dropout_rate=0.2,repeats =3,train = False)



  return x,y


def create_sv_model():
  inputs = tf.keras.layers.Input(shape = (13,3496))
  output1,output2 = create_Jasper(inputs,5,1)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
  model_sv = keras.Model(inputs=inputs, outputs =output2 )
  model_sv.summary()
  plot_model(model_sv, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  return model_sv

if __name__ ==  "main":
  speaker_verification_model = create_sv_model()

