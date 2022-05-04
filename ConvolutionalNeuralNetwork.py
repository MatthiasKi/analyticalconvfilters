import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Activation, Input, Flatten, Conv2D
from timeit import default_timer as timer
import os

from PseudoRegressorKeras import RandomOrthogonal

def get_activation(activation_name):
    if activation_name == "lrelu":
        return LeakyReLU(alpha=2.)
    else:
        return Activation(activation=activation_name)

class TimingCallback(keras.callbacks.Callback):
    def __init__(self, log_filename):
        self.log_filename = log_filename

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
        with open(self.log_filename, "a+") as f:
            f.write("Begin Time of Epoch " + str(epoch) + ": " + str(self.starttime) + "\n")

    def on_epoch_end(self, epoch, logs={}):
        with open(self.log_filename, "a+") as f:
            f.write("Epoch " + str(epoch) + ": " + str(timer()-self.starttime) + " seconds\n")
            f.write("End Time of Epoch " + str(epoch) + ": " + str(timer()) + "\n")

class AccuracyCallback(keras.callbacks.Callback):
    # cechkout https://github.com/keras-team/keras/issues/2548
    def __init__(self, log_filename, valX, valY):
        self.log_filename = log_filename
        self.valX = valX
        self.valY = valY

    def on_epoch_end(self, epoch, logs={}):
        with open(self.log_filename, "a+") as f:
            f.write("Val Acc Start Time of Epoch " + str(epoch) + ": " + str(timer()) + "\n")

        # TODO NOTE this does only work for categorical data
        val_acc = np.sum(np.argmax(self.valY, axis=1) == np.argmax(self.model.predict(self.valX), axis=1))/len(self.valY)
        
        with open(self.log_filename, "a+") as f:
            f.write("Val Acc at the end of Epoch " + str(epoch) + ": " + str(val_acc) + "\n")
            f.write("Val Acc End Time of Epoch " + str(epoch) + ": " + str(timer()) + "\n")

class ConvolutionalNeuralNetwork():
    def __init__(self,n_conv_layers, conv_activation, n_conv_filters, input_shape, n_dense_layers, n_dense_neurons, dense_activation, kernel_init="glorot_uniform", num_classes_for_categorical=1, epochs=10000, strides=(3,3)):
        if kernel_init == "random_orthogonal":
            kernel_initializer = RandomOrthogonal()
        else:
            kernel_initializer = kernel_init

        self.epochs = epochs

        # Note that by reusing "act", the layer gets shared across the model and hence is only once present in self.model.layers => always intantiate new layers
        self.model = Sequential()
        self.model.add(Input(shape=input_shape))
        for i in range(n_conv_layers):
            self.model.add(Conv2D(n_conv_filters, kernel_size=(3,3), strides=strides, padding="valid", kernel_initializer=kernel_initializer, bias_initializer='zeros'))
            self.model.add(get_activation(conv_activation))
        self.model.add(Flatten())
        for i in range(n_dense_layers):
            self.model.add(Dense(n_dense_neurons,kernel_initializer=kernel_initializer,bias_initializer='zeros'))
            self.model.add(get_activation(dense_activation))
        self.model.add(Dense(num_classes_for_categorical))
        
        self.model.compile(loss='mean_squared_error',optimizer='adam')

    def fit(self,trainX,trainY, valX, valY, iteration=0):
        if not os.path.isdir("data"):
            os.mkdir("data")

        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        timing_callback = TimingCallback(log_filename="data/Iteration" + str(iteration) + ".txt")
        accuracy_callback = AccuracyCallback(log_filename=("data/Iteration" + str(iteration) + ".txt"), valX=valX, valY=valY)
        self.model.fit(trainX,trainY,batch_size=32,validation_data=(valX,valY),epochs=self.epochs,callbacks=[es_callback, timing_callback, accuracy_callback], verbose=0)

    def predict(self,X):
        return self.model.predict(X)