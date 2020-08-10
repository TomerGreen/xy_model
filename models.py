import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Input, Conv1D, Add, BatchNormalization, Activation, Dense, ZeroPadding1D, \
    Lambda, AveragePooling1D
from tensorflow.python.keras.models import Model
from layers import SliceRepetitions


ENCODER_KERNEL_SIZE = 1
DEFAULT_EMBEDDING_LENGTH = 1
CONV_BLOCK_FILTERS = [32, 32, 32]
CONV_BLOCK_KERNELS_SIZES = [3, 3, 3]
CONV_BLOCK_STRIDES = [1, 1, 1]


def apply_conv_block(x, conv_layers, bn_layers, training=True):
    assert len(conv_layers) == len(bn_layers)
    # TODO - consider ResNet.
    for i in range(len(conv_layers)):
        x = ZeroPadding1D(tuple([int((conv_layers[i].kernel_size[0] - 1) / 2)] * 2))(x)
        x = conv_layers[i](x)
        x = bn_layers[i](x, training=training)
        x = Activation('relu')(x)
    return x


def get_correlation_model(num_spins, num_reps=3, training=True, embedding_length=DEFAULT_EMBEDDING_LENGTH):
    """
    Implements a convolutional model that finds z_spin correlations from XY model data. The function constructs a
    convolution block and applies it repeatedly on the input, pooling by half after each time, until the input is at
    a predetermined size. It then applies a decoder block to obtain the correlation as a scalar.
    :param num_spins: the number of spins in the spin chain
    :param num_reps: an odd integer. the number of times the input repeats itself, to imitate periodic boundary
    conditions.
    :param training: whether the model is in training or inference mode.
    :return: a model whose input is of shape(batch_size, num_spins, 3) and output is of shape (batch_size,)
    """
    inputs = Input(shape=(num_spins * num_reps, 3))
    x = inputs
    x = ZeroPadding1D(tuple([int((ENCODER_KERNEL_SIZE - 1) / 2)] * 2))(x)
    x = Conv1D(CONV_BLOCK_FILTERS[0], ENCODER_KERNEL_SIZE)(x)
    conv_layers, bn_layers = [], []
    for i in range(len(CONV_BLOCK_FILTERS)):
        conv_layers.append(Conv1D(filters=CONV_BLOCK_FILTERS[i],
                                  kernel_size=CONV_BLOCK_KERNELS_SIZES[i],
                                  strides=CONV_BLOCK_STRIDES[i],
                                  padding='valid',
                                  use_bias=False))
        bn_layers.append(BatchNormalization())
    num_conv_blocks = int(np.log2(num_spins / embedding_length))
    for i in range(num_conv_blocks):
        x = apply_conv_block(x, conv_layers, bn_layers, training=training)
        x = AveragePooling1D(pool_size=2, padding='valid')(x)
    # x = Conv1D(filters=1, kernel_size=1)(x)
    x = Dense(1)(x)
    x = Lambda(tf.squeeze, arguments={'axis': 2})(x)
    x = SliceRepetitions(final_size=DEFAULT_EMBEDDING_LENGTH, num_reps=num_reps)(x)
    # TODO - Consider intermediate dense layer.
    # x = Activation('sigmoid')(x)
    # x = Dense(1, activation='sigmoid')(x)
    # x = Lambda(lambda x: 2 * x - 1)(x)
    # x = Lambda(tf.squeeze, arguments={'axis': 1})(x)
    return Model(inputs=inputs, outputs=x)



if __name__ == '__main__':
    model = get_correlation_model()
    model.summary(100)
    model.save('./test_model.h5')
    # x = np.array([[[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]]])
    x = np.array([[[0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 1, 1]], [[1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1]]])
    x = x.astype(np.float32)
    y = np.array([1, 0])
    model.compile(optimizer='adam', loss='mse')
    print(x.shape)
    model.fit(x, y, epochs=100)
    y_pred = model.predict(x)
    print(y_pred)
