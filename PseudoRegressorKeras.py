import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LeakyReLU, Activation, Conv2D, Flatten, Input
from sklearn import linear_model
from tensorflow.keras.initializers import Initializer
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import scipy
from skimage.util import view_as_blocks

def get_activation(activation_name):
    if activation_name == "lrelu":
        return LeakyReLU(alpha=2.)
    else:
        return Activation(activation=activation_name)

# Inverse Activation Functions
def arctanh(X):
    X_clipped = np.clip(X, a_min=-0.99, a_max=0.99)
    return np.arctanh(X_clipped)
def inv_lrelu(X):
    X_act = X.copy()
    X_act[X < 0.] = 0.5 * X_act[X < 0.]
    return X_act
def inv_sigmoid(X):
    X_clipped = np.clip(X, a_min=0.01, a_max=0.99)
    return np.log(X_clipped / (1.-X_clipped))

def get_mod_boundary(length):
    if np.mod(length,3) == 0:
        return length
    else:
        return -np.mod(length,3)

def adjust_conv_input_shape(input_shape):
    adj_first_index = np.mod(input_shape[0], 3)
    adj_second_index = np.mod(input_shape[1], 3)
    return (input_shape[0]-adj_first_index, input_shape[1]-adj_second_index, input_shape[2])

def get_inv_activation(activation_name):
    if activation_name == "tanh":
        return arctanh
    elif activation_name == "lrelu":
        return inv_lrelu
    elif activation_name == "sigmoid":
        return inv_sigmoid
    else:
        raise Exception("Activation unknown")
    
def mse(y, y_hat):
    return np.mean(np.square(y.flatten() - y_hat.flatten()))

def random_cond1_matrix(shape):
    if len(shape) != 2:
        raise Exception("Matrix Shape is incompatible with random_cond1_matrix")

    if shape[0] >= shape[1]:
        mat = scipy.stats.ortho_group.rvs(shape[0])
        mat = mat[:, :shape[1]]
    else:
        mat = scipy.stats.ortho_group.rvs(shape[1])
        mat = mat[:shape[0], :]

    return mat

def flatten_outputs(mat_in, copy=True):
    if copy == True:
        mat = mat_in.copy()
    else:
        mat = mat_in

    return np.reshape(mat, (mat.shape[0], -1))

def reshape_flattened_to_output(mat_in, output_shape, copy=True):
    if copy == True:
        mat = mat_in.copy()
    else:
        mat = mat_in

    if np.mod(mat.size,np.prod(output_shape)) != 0:
        raise Exception("Number of Input Images could not be inferred")
    nb_images = int(mat.size/np.prod(output_shape))

    return np.reshape(mat, (nb_images, *output_shape))

def reshape_output_matrix_to_ls_shape(mat_in, copy=True):
    if copy == True:
        mat = mat_in.copy()
    else:
        mat = mat_in

    mat = np.transpose(mat, [1,2,0,3])
    mat = np.reshape(mat, (-1, mat.shape[2], mat.shape[3]), order='C')
    mat = np.transpose(mat, [1,0,2])

    return np.concatenate([im for im in mat], axis=0)

def reshape_ls_shape_to_output_matrix(mat_in, image_shape, copy=True):
    if copy == True:
        mat = mat_in.copy()
    else:
        mat = mat_in

    patches_per_image = int(image_shape[0]*image_shape[1])

    # Extract single images from the batch-combined matrix
    ims = np.reshape(mat, (-1, patches_per_image, mat.shape[1]), order='C')

    ims_reshaped = [np.expand_dims(np.reshape(im, (image_shape[0], image_shape[1], mat.shape[1]), order='C'), 0) for im in ims]

    return np.concatenate(ims_reshaped, axis=0)

def reshape_ls_coefs_to_keras_weights(mat_in, copy=True):
    if copy == True:
        mat = mat_in.copy()
    else:
        mat = mat_in
    
    if np.mod(mat.shape[0], 9) != 0:
        raise Exception("No match for the number of filters found")
    n_filters = int(mat.shape[0] / 9) # It is assumed that the kernel_size is 3!

    mat = np.transpose(mat, [1,0])
    mat = np.reshape(mat, (mat.shape[0], n_filters, 3, 3))
    mat = np.transpose(mat, [2, 3, 1, 0])
    return mat

def reshape_keras_weights_to_ls_coefs(mat_in, copy=True):
    if copy == True:
        mat = mat_in.copy()
    else:
        mat = mat_in

    mat = np.transpose(mat, [3,2,0,1])
    mat = np.reshape(mat, (mat.shape[0], mat[0].size))
    mat = np.transpose(mat, [1,0])
    return mat

def reshape_block_matrix_to_ls_shape(mat_in, copy=True):
    if copy == True:
        mat = mat_in.copy()
    else:
        mat = mat_in
    mat = np.reshape(mat, (mat.shape[0], mat.shape[1], mat.shape[2]*mat.shape[3]), order='C')
    mat = np.reshape(mat, (mat.shape[0] * mat.shape[1], mat.shape[2]), order='F')
    return mat

def reshape_input_matrix_to_ls_shape(mat_in, copy=True):
    if copy == True:
        mat = mat_in.copy()
    else:
        mat = mat_in

    # Concatenate images in the y-axis - needed to handle batches
    mat = np.concatenate([np.transpose(im, [1,0,2]) for im in mat], axis=1)

    # Represent as blocks
    block_matrices = []
    for dim in range(mat.shape[2]):
        curr_blocks = view_as_blocks(mat[:,:,dim], block_shape=(3,3)) # The function view_as_windows could also handle strides which do not match the kernel shape
        curr_blocks = np.transpose(curr_blocks, [0,1,3,2])
        curr_blocks = reshape_block_matrix_to_ls_shape(curr_blocks)
        block_matrices.append(curr_blocks)
    mat = np.concatenate(block_matrices, axis=1)

    return mat

def sub_reshape_ls_shape_to_input_matrix(mat, n_filters):
    img_len = int(np.sqrt(mat.shape[0])) # It is assumed that the image is square!
    mat = np.reshape(mat, (mat.shape[0], 3, 3, n_filters), order='F')
    mat = np.transpose(mat, [0,2,1,3])
    mat = np.reshape(mat, (img_len, img_len, 3, 3, n_filters), order='C')
    mat = np.transpose(mat, [4, 0, 1, 2, 3])
    raw_sub_bmats = [[[sub_mat for sub_mat in sub] for sub in dim_mat] for dim_mat in mat]
    sub_bmats = [np.expand_dims(np.bmat(mat), -1) for mat in raw_sub_bmats]
    mat = np.concatenate(sub_bmats, axis=-1)
    return mat

def reshape_ls_shape_to_input_matrix(mat_in, image_shape, copy=True):
    if copy == True:
        mat = mat_in.copy()
    else:
        mat = mat_in

    if np.mod(mat.shape[1], 9) != 0:
        raise Exception("No match for the number of filters found")
    n_filters = int(mat.shape[1] / 9) # It is assumed that the kernel_size is 3!

    patches_per_image = int(image_shape[0]*image_shape[1] / 9)

    # Extract single images from the batch-combined matrix
    ims = np.reshape(mat, (-1, patches_per_image, mat.shape[1]))

    # Process each image
    ims_reshaped = [np.expand_dims(sub_reshape_ls_shape_to_input_matrix(mat, n_filters),0) for mat in ims]

    return np.concatenate(ims_reshaped, axis=0)


class RandomOrthogonal(Initializer):
    # Initializes with an orthogonal weight matrix with random base vectors (i.e. condition number is 1)

    def __call__(self, shape, dtype=None):
        return K.constant(random_cond1_matrix(shape))

class ConvolutionalPseudoRegressor():
    def __init__(self,n_conv_layers, conv_activation, n_conv_filters, input_shape, n_dense_layers, n_dense_neurons, dense_activation, kernel_init="glorot_uniform", first_trainable_layer=0, num_classes_for_categorical=1, alpha=1e-3, verbose=False, strides=(3,3)):
        # NOTE that in this proof-of-concept implementation, all kernel shapes are set to (3,3) and strides are 3 in all directions

        if strides != (3,3) and 2*n_conv_layers + 1 + 2*n_dense_layers != first_trainable_layer:
            raise Exception("In order to train the convolutional filters of the CELM, the size of the kernel must match the strides")

        self.alpha = alpha
        self.verbose = verbose

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

        if n_conv_layers == 0:
            self.last_conv_activation_layer_i = -1
        else:
            self.last_conv_activation_layer_i = max([i for i in range(len(self.model.layers)) if isinstance(self.model.layers[i], Conv2D)]) + 1
        self.layer_indices = [i for i, layer in enumerate(self.model.layers) if (isinstance(layer, Dense) or isinstance(layer, Conv2D)) and i >= first_trainable_layer]

    def fit(self, trainX, trainY_raw, max_retraining_epochs_after_fitting=0, valX=None, valY_raw=None):
        # NOTE that biases are fitted by using the interceptions of the linear model; do not try to fit them explicitly as this is bad for the condition number of the feature matrix
        # NOTE: valX and valY are only used for retraining after fitting!
        if max_retraining_epochs_after_fitting > 0 and (valX is None or valY_raw is None):
            raise Exception("Validation X and Y must be given if retraining after fitting is used")

        # Make sure that trainY is two-dimensional
        if len(trainY_raw.shape) == 1:
            trainY = np.expand_dims(trainY_raw, -1)
        else:
            trainY = trainY_raw
        if self.verbose:
            print("Error before training: " + str(mse(trainY, self.model.predict(trainX))))

        # Forward pass is self.layer_indices, backward pass would be self.layer_indices[::-1]
        for layer_index in self.layer_indices:
            # Calculate the activation for this layer
            activation = self.activation_at_layer(layer_index, trainX)

            # Calculate the target activation for this layer
            target_activation = self.__desired_activation_at_layer(layer_index, trainY)

            if isinstance(self.model.layers[layer_index], Dense):
                # Solve the system of equations
                reg = linear_model.Ridge(alpha=self.alpha)
                reg.fit(activation, target_activation)

                # Set the model weights accordingly
                # NOTE: The linear model would return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
                self.model.layers[layer_index].set_weights([reg.coef_.T, reg.intercept_])
            elif isinstance(self.model.layers[layer_index], Conv2D):
                # It is assumed that the inputs are of the shape (lxlxd)
                if len(self.model.layers[layer_index].input_shape) != 4 or self.model.layers[layer_index].input_shape[1] != self.model.layers[layer_index].input_shape[2]:
                    raise Exception("Input shape does not fit")

                # Bring the inputs and outputs into appropriate shapes
                # Target shape should be right per default
                activation = activation[:, :(target_activation.shape[1]*3), :(target_activation.shape[2]*3), :]

                # Represent inputs in the right shape
                activation = reshape_input_matrix_to_ls_shape(activation)

                # Reshape outputs to LS shape
                target_activation = reshape_output_matrix_to_ls_shape(target_activation)

                # Build the regression model
                reg = linear_model.Ridge(alpha=self.alpha)
                reg.fit(activation, target_activation)

                # Reshape the weights such that it fits to the keras representation, activation has the shape (nxr), filter has the shape (rxk)
                kernel_weights = reshape_ls_coefs_to_keras_weights(reg.coef_.T)

                # Set the weights accordingly
                self.model.layers[layer_index].set_weights([kernel_weights, reg.intercept_])
            else:
                raise Exception("Not recognized layer type in network")

            if self.verbose:
                print("Update: " + str(mse(trainY, self.model.predict(trainX))))

        if max_retraining_epochs_after_fitting > 0:
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            self.model.compile(loss='mean_squared_error',optimizer=optimizer)
            es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            self.model.fit(trainX,trainY_raw,batch_size=32,validation_data=(valX,valY_raw),epochs=max_retraining_epochs_after_fitting,callbacks=[es_callback], verbose=0)

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

    def __desired_activation_at_layer(self, idx, y_target):
        # NOTE: In this function it is implicitly assumed that the biases are initialized with zeros, i.e. they are not considered when the y's are backpropagated
        prop = y_target.copy()
        for layer_i, layer in enumerate(self.model.layers[idx+1:][::-1]): # NOTE: Inputs are NOT listed in the model.layers! (NOTE: This is because we are using a Sequential Model, in Functional Models, the Input layers are listed!); Need the idx+1 because I would like to have the OUTPUTS at the regarded layer (which are the desired INPUTS of the subsequent layer)
            curr_layer_i = len(self.model.layers) - 1 - layer_i
            if isinstance(layer, Dense):
                W = layer.get_weights()[0]
                reg = linear_model.Ridge(alpha=self.alpha, fit_intercept=False) # NOTE that we dont need fit_intercept here.
                reg.fit(W.T, prop.T)
                prop = reg.coef_ # NOTE: Do not need to transpose here, as the target value is again transposed
            elif isinstance(layer, Conv2D):
                # NOTE that in this step it might cause problems if the stride length does not match the kernel!
                W_ls = reshape_keras_weights_to_ls_coefs(layer.get_weights()[0])
                prop_ls = reshape_output_matrix_to_ls_shape(prop)

                reg = linear_model.Ridge(alpha=self.alpha, fit_intercept=False) # NOTE that we dont need fit_intercept here.
                reg.fit(W_ls.T, prop_ls.T)

                prop = reshape_ls_shape_to_input_matrix(reg.coef_, adjust_conv_input_shape(layer.input_shape[1:]))
            elif isinstance(layer, Activation):
                # NOTE that here it is assumed that the model is split into a first part with only Conv-Layers and a second part with only Dense layers
                if curr_layer_i > self.last_conv_activation_layer_i:
                    prop = self.dense_inv_act(prop)
                else:
                    prop = self.conv_inv_act(prop)
            elif isinstance(layer, Flatten):
                prop = reshape_flattened_to_output(prop, self.model.layers[curr_layer_i-1].output_shape[1:])
        return prop

    def predict(self,X):
        return self.model.predict(X)

if __name__ == "__main__":
    # Tests
    test_mat = np.arange(8*12*12*3, dtype="float32").reshape((8,12,12,3))

    reshaped_to_ls = reshape_input_matrix_to_ls_shape(test_mat)
    backshaped_to_input = reshape_ls_shape_to_input_matrix(reshaped_to_ls, test_mat[0].shape)

    if not np.all(backshaped_to_input == test_mat):
        raise Exception("Input Reshape Unit Test failed")

    backshaped_to_ls = reshape_output_matrix_to_ls_shape(test_mat)
    reshaped_to_output = reshape_ls_shape_to_output_matrix(backshaped_to_ls, test_mat[0].shape)

    if not np.all(reshaped_to_output == test_mat):
        raise Exception("Output Reshape Unit Test failed")

    test_ls_weights = np.arange(9*5).reshape(9,5)

    reshaped_to_keras = reshape_ls_coefs_to_keras_weights(test_ls_weights)
    backshaped_keras_to_ls = reshape_keras_weights_to_ls_coefs(reshaped_to_keras)

    if not np.all(backshaped_keras_to_ls == test_ls_weights):
        raise Exception("Keras Weight Reshape Unit Test failed")

    test_inp = Input(shape=test_mat[0].shape)
    test_layer = Conv2D(10, kernel_size=(3,3),strides=(3,3), bias_initializer="zeros")(test_inp)
    test_model = Model(inputs=test_inp, outputs=test_layer)
    
    keras_weights = test_model.layers[1].get_weights()[0]
    test_model.layers[1].set_weights([np.reshape(np.arange(np.prod(keras_weights.shape)), keras_weights.shape), test_model.layers[1].get_weights()[1]])
    keras_weights = test_model.layers[1].get_weights()[0]
    test_output = test_model(test_mat).numpy()

    backshaped_weights_to_ls = reshape_keras_weights_to_ls_coefs(keras_weights)
    reshaped_weights_to_keras = reshape_ls_coefs_to_keras_weights(backshaped_weights_to_ls)

    ls_input = reshape_input_matrix_to_ls_shape(test_mat)
    ls_output = ls_input @ backshaped_weights_to_ls
    output_reshaped = reshape_ls_shape_to_output_matrix(ls_output, test_model.layers[1].output_shape[1:])

    if not np.all(test_output == output_reshaped):
        raise Exception("Keras Function Unit Test failed")

    flattened = flatten_outputs(test_mat)
    reshaped_from_flattened = reshape_flattened_to_output(flattened, test_mat.shape[1:])

    if not np.all(reshaped_from_flattened == test_mat):
        raise Exception("Flatten Unit Test failed")

    test_inp2 = Input(shape=test_mat[0].shape)
    test_layer2 = Flatten()(test_inp2)
    test_model2 = Model(inputs=test_inp2, outputs=test_layer2)

    keras_flattened = test_model2.predict(test_mat)

    if not np.all(flattened == keras_flattened):
        raise Exception("Flatten vs Keras Flatten unit Test failed")

    valid_activation = np.random.uniform(-1,1, size=(8,9,11,3))
    valid_activation = valid_activation[:, :get_mod_boundary(valid_activation.shape[1]), :get_mod_boundary(valid_activation.shape[2]), :]

    test_inp3 = Input(shape=valid_activation[0].shape)
    test_layer3 = Conv2D(10, kernel_size=(3,3),strides=(3,3), bias_initializer="zeros", kernel_initializer="glorot_uniform", padding="valid")(test_inp3)
    test_model3 = Model(inputs=test_inp3, outputs=test_layer3)
    valid_keras = test_model3.predict(valid_activation)

    keras_weights = test_model3.layers[1].get_weights()[0]
    ls_input = reshape_input_matrix_to_ls_shape(valid_activation)
    ls_weights = reshape_keras_weights_to_ls_coefs(keras_weights)
    ls_output = ls_input @ ls_weights
    valid_output = reshape_ls_shape_to_output_matrix(ls_output, test_model3.output_shape[1:])

    print(np.max(np.abs(valid_output-valid_keras)))
    print(np.mean(np.abs(valid_output-valid_keras)))