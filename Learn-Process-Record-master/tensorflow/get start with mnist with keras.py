# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 18:34:05 2018

@author: xzc
"""
# import the tensorflow module
import tensorflow as tf

# prepare the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test /255.0

# construct the model
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation = tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
        ])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 5)
model.evaluate(x_test, y_test)


# the official tutorial

# to get started, umport tf.keras as part of tensorflow setup
import tensorflow as tf
from tensorflow import keras

# build a simple model
model = keras.Sequential()
# adds a densely-connected layer with 64 units to the model
model.add(keras.layers.Dense(64, activation='relu'))
# add another
model.add(keras.layers.Dense(64, activation='relu'))
# add a softmax layer with 10 output units
model.add(keras.layers.Dense(10, activation='softmax'))

# tf.keras.layers.Dense

# create a sigmoid layer
layers.Dense(64, activation = 'sigmoid')

# input numpy data
import numpy as np
data = np.random.random(1000, 32)
labels = np.random.random((1000, 0))

model.fit(data, labels, epochs=10, batch_size=32)

# use a functional API to build a simple, fully-connected network
inputs = keras.Input(shape = (32,))
# returns a placeholder tensor

# a layer instances is callable on a tensor, and returns a tensor
x = keras.kayers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)

predictions = keras.layers.Dense(10, activation = 'softmax')(x)

# instantiate the model given inputs and outputs
model = keras.Model(inputs=inputs, outputs=predictions)

# the compile step specifies the training configuration
model.compile(optimizer = tf.train.RMSPropOptimizer(0.001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
# trains for 5 epochs
model.fit(data, labels, batch_size = 32, epochs = 5)

# a subclassed tf.keras.Model using a custom forward pass
class MyModel(keras.Model):
    
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__(name = 'my_model')
        self.num_classes = num_classes
        # define your layers here
        self.dense_1 = keras.layers.Dense(32, activation = 'relu')
        self.dense_2 = keras.layers.Dense(num_classes, activation='sigmoid')
    
    def call(self, inputs):
        # define your forward pass here
        # using layers you previously
        x = self.dense_1(inputs)
        return self.dense_2(x)
    
    def compute_output_shape(self, input_shape):
        # you need to override this function if you want to use the subclassed model
        # as part of a functional-styled model
        # otherwise, this method is optional
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

# instantiate the subclassed model
model = MyModel(num_classes = 10)

# the compile step specifies the training configuration
model.compile(optimizer = tf.train.RMSPropOptimizer(0.001),
              loss = 'categorical_crossentrophy',
              metrics = ['accuracy'])

# trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)

# custom layers
# create a custom layer by subclassing tf.keras.layers.Layer

class MyLayer(keras.layers.Layer):
    
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init(**kwargs)
        
    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        # create a trainable weight variable for this layer
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = shape,
                                      initializer = 'uniform',
                                      trainable = True)
        # be sure to call this at the end
        super(MyLayer, self).build(input_shape)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.kernal)
    
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)
    
    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# create a model using the custom layer
model = keras.Sequential([MyLayer(10),
                         keras.layers.Activation('softmax')])

# the compile step specifies the training configuration
model.compile(optimizer = tf.train.RMSPropOptimizer(0.001),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
# trains for 5 epochs
model.fit(data, targets, batch_size = 32, epochs = 5)





















