from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.datasets import mnist, cifar10, cifar100, fashion_mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from PseudoRegressorKeras import ConvolutionalPseudoRegressor, mse
from ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from KMeans_ELM import KMeans_ELM
from RandomPatch_ELM import RandomPatch_ELM

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# Hyperparameters
n_conv_layers = 2
n_dense_layers = 0
n_conv_filters = 50
n_dense_neurons = 200
conv_activation = "lrelu"
dense_activation = "lrelu"
iterations = 1
first_trainable_layer = 0 # Only used for PseudoRegressor, Note that there are activation layers in between!
kernel_initializer = "glorot_uniform" # Only used for PseudoRegressor (glorot_uniform | random_orthogonal)
dataset = "CIFAR10" # "MNIST", "CIFAR10", "CIFAR100" or "FashionMNIST"
use_categorical_labels = True
scale = True
nn_epochs = 300
debug_run = True # If true, only a small subset of the training / test set will be used
# ---

if dataset == "MNIST":
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()
elif dataset == "CIFAR10":
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = cifar10.load_data()
elif dataset == "CIFAR100":
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = cifar100.load_data()
elif dataset == "FashionMNIST":
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = fashion_mnist.load_data()
else:
    raise Exception("Dataset not recognized")

if len(X_train_raw[0].shape) == 2:
    X_train = np.expand_dims(X_train_raw, -1).copy()
    X_test = np.expand_dims(X_test_raw, -1).copy()
elif len(X_train_raw[0].shape) == 3:
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
else:
    raise Exception("Input Shape wrong")
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

if scale:
    x_max = np.max(X_train)
    x_min = np.min(X_train)
    X_train = ((X_train - x_min) / (x_max - x_min)) - 0.5
    X_test = ((X_test - x_min) / (x_max - x_min)) - 0.5

if use_categorical_labels:
    num_classes_for_categorical = len(np.unique(y_train_raw))

    y_train_raw = to_categorical(y_train_raw, num_classes_for_categorical)
    y_test_raw = to_categorical(y_test_raw, num_classes_for_categorical)

    y_train = y_train_raw.copy()
    y_test = y_test_raw.copy()
else:
    y_train_raw = np.expand_dims(y_train_raw, -1)
    y_test_raw = np.expand_dims(y_test_raw, -1)
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw)
    y_test = y_scaler.transform(y_test_raw)

    num_classes_for_categorical = 1

if debug_run:
    X_train = X_train[:1000]
    X_test = X_test[:1000]
    y_train = y_train[:1000]
    y_test = y_test[:1000]
    y_test_raw = y_test_raw[:1000]
    y_train_raw = y_train_raw[:1000]

model_names = ["Pseudo", "ELM", "NN", "KMeans-ELM", "RandomPatch-ELM"]
train_errors = [[] for i in range(len(model_names))]
test_errors = [[] for i in range(len(model_names))]
train_accuracies = [[] for i in range(len(model_names))]
test_accuracies = [[] for i in range(len(model_names))]
for _ in range(iterations):
    pseudo_reg = ConvolutionalPseudoRegressor(n_conv_layers=n_conv_layers, conv_activation=conv_activation, n_conv_filters=n_conv_filters, input_shape=X_train[0].shape, n_dense_layers=n_dense_layers, n_dense_neurons=n_dense_neurons, dense_activation=dense_activation, kernel_init=kernel_initializer, first_trainable_layer=first_trainable_layer, num_classes_for_categorical=num_classes_for_categorical)
    pseudo_reg.fit(X_train, y_train)

    elm_first_trainable_layer = 2*n_conv_layers + 1 + 2*n_dense_layers # Conv Layers + Flatten + Dense Layers (each with activation)
    elm = ConvolutionalPseudoRegressor(n_conv_layers=n_conv_layers, conv_activation=conv_activation, n_conv_filters=n_conv_filters, input_shape=X_train[0].shape, n_dense_layers=n_dense_layers, n_dense_neurons=n_dense_neurons, dense_activation=dense_activation, kernel_init=kernel_initializer, first_trainable_layer=elm_first_trainable_layer, num_classes_for_categorical=num_classes_for_categorical)
    elm.fit(X_train, y_train)

    nn = ConvolutionalNeuralNetwork(n_conv_layers=n_conv_layers, conv_activation=conv_activation, n_conv_filters=n_conv_filters, input_shape=X_train[0].shape, n_dense_layers=n_dense_layers, n_dense_neurons=n_dense_neurons, dense_activation=dense_activation, kernel_init=kernel_initializer, num_classes_for_categorical=num_classes_for_categorical, epochs=nn_epochs)
    nn.fit(X_train, y_train, X_train, y_train) # TODO: Note that the train set is treated as validation set here!

    kmeans_elm = KMeans_ELM(n_conv_layers=n_conv_layers, conv_activation=conv_activation, n_conv_filters=n_conv_filters, input_shape=X_train[0].shape, n_dense_layers=n_dense_layers, n_dense_neurons=n_dense_neurons, dense_activation=dense_activation, kernel_init=kernel_initializer, num_classes_for_categorical=num_classes_for_categorical)
    kmeans_elm.fit(X_train, y_train)

    randompatch_elm = RandomPatch_ELM(n_conv_layers=n_conv_layers, conv_activation=conv_activation, n_conv_filters=n_conv_filters, input_shape=X_train[0].shape, n_dense_layers=n_dense_layers, n_dense_neurons=n_dense_neurons, dense_activation=dense_activation, kernel_init=kernel_initializer, num_classes_for_categorical=num_classes_for_categorical)
    randompatch_elm.fit(X_train, y_train)

    models = [pseudo_reg, elm, nn, kmeans_elm, randompatch_elm]

    if use_categorical_labels:
        for model_i, (model, model_name) in enumerate(zip(models, model_names)):
            train_errors[model_i].append(mse(y_train_raw, model.predict(X_train)))
            test_errors[model_i].append(mse(y_test_raw, model.predict(X_test)))

            train_accuracies[model_i].append(np.sum(np.argmax(y_train_raw, axis=1) == np.argmax(model.predict(X_train), axis=1))/len(y_train_raw))
            test_accuracies[model_i].append(np.sum(np.argmax(y_test_raw, axis=1) == np.argmax(model.predict(X_test), axis=1))/len(y_test_raw))
    else:
        for model_i, (model, model_name) in enumerate(zip(models, model_names)):
            train_errors[model_i].append(mse(y_train_raw, y_scaler.inverse_transform(model.predict(X_train))))
            test_errors[model_i].append(mse(y_test_raw, y_scaler.inverse_transform(model.predict(X_test))))

            train_accuracies[model_i].append(np.sum(y_train_raw == np.round(model.predict(X_train))) / len(y_train_raw))
            test_accuracies[model_i].append(np.sum(y_test_raw == np.round(model.predict(X_test))) / len(y_test_raw))

for model_name, train_error, test_error, train_accuracy, test_accuracy in zip(model_names, train_errors, test_errors, train_accuracies, test_accuracies):
    print(model_name + " Error: \t " + str(np.mean(test_error)) + " +- " + str(np.std(test_error)) + " \t [Train: \t " + str(np.mean(train_error)) + " +- " + str(np.std(train_error)) + "]")
    print(model_name + " Acc.: \t " + str(np.mean(test_accuracy)) + " +- " + str(np.std(test_accuracy)) + " \t [Train: \t " + str(np.mean(train_accuracy)) + " +- " + str(np.std(train_accuracy)) + "]")