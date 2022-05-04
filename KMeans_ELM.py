import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D
from sklearn import linear_model
from sklearn.cluster import KMeans

from PseudoRegressorKeras import get_activation, get_inv_activation, RandomOrthogonal, reshape_input_matrix_to_ls_shape, reshape_ls_coefs_to_keras_weights, get_mod_boundary

class KMeans_ELM():
    def __init__(self,n_conv_layers, conv_activation, n_conv_filters, input_shape, n_dense_layers, n_dense_neurons, dense_activation, kernel_init="glorot_uniform", num_classes_for_categorical=1, alpha=1e-3, strides=(3,3)):
        self.alpha = alpha
        self.n_conv_filters = n_conv_filters

        self.conv_inv_act = get_inv_activation(conv_activation)
        self.dense_inv_act = get_inv_activation(dense_activation)

        if kernel_init == "random_orthogonal":
            kernel_initializer = RandomOrthogonal()
        else:
            kernel_initializer = kernel_init

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

    def activation_at_layer(self, idx, X):
        if idx == 0:
            # For idx = 0, the activations are the inputs themself
            activation = X.copy()
        else:
            # Clone the model until the desired layer
            cloned_model = Sequential()
            for layer in self.model.layers[:idx]:
                cloned_model.add(layer)
            
            # The activations are the predictions of the cloned model
            activation = cloned_model.predict(X)

        # Return the activations
        return activation

    def fit(self,trainX,trainY_raw):
        # Make sure that trainY is two-dimensional
        if len(trainY_raw.shape) == 1:
            trainY = np.expand_dims(trainY_raw, -1)
        else:
            trainY = trainY_raw

        conv_layers = [i for i, layer in enumerate(self.model.layers) if isinstance(layer, Conv2D)]
        
        for layer_index in conv_layers:
            activation = self.activation_at_layer(layer_index, trainX)

            # Crop activation such that the shape fits to the reshaping
            activation = activation[:, :get_mod_boundary(activation.shape[1]), :get_mod_boundary(activation.shape[2]), :]
            reshaped_activations = reshape_input_matrix_to_ls_shape(activation)

            # Get KMeans cluster centers
            kmeans = KMeans(n_clusters=self.n_conv_filters)
            kmeans.fit(reshaped_activations)
            weights_raw = kmeans.cluster_centers_.T
            weights = reshape_ls_coefs_to_keras_weights(weights_raw)
            
            # Leave the bias as it is
            bias = self.model.layers[layer_index].bias.numpy()
            self.model.layers[layer_index].set_weights([weights, bias]) 

        dense_layers = [i for i, layer in enumerate(self.model.layers) if isinstance(layer, Dense)]
        # Fit last Dense layer ELM like
        features = self.activation_at_layer(dense_layers[-1], trainX)
        reg = linear_model.Ridge(alpha=self.alpha)
        reg.fit(features, trainY)
        self.model.layers[dense_layers[-1]].set_weights([reg.coef_.T, reg.intercept_])

    def predict(self,X):
        return self.model.predict(X)