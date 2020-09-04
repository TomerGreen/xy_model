import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Input, Conv1D, Add, BatchNormalization, Activation, Dense, ZeroPadding1D, \
    Lambda, AveragePooling1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from layers import SliceRepetitions


ENCODER_KERNEL_SIZE = 1
DEFAULT_EMBEDDING_LENGTH = 1
CONV_BLOCK_FILTERS = [32, 32, 32]
CONV_BLOCK_KERNELS_SIZES = [3, 3, 3]
CONV_BLOCK_STRIDES = [1, 1, 1]
REG = 1e-4


def apply_conv_block(x, conv_layers, bn_layers, block_num, training=True):
    assert len(conv_layers) == len(bn_layers)
    # TODO - consider ResNet.
    for i in range(len(conv_layers)):
        x = ZeroPadding1D(tuple([int((conv_layers[i].kernel_size[0] - 1) / 2)] * 2),
                          name='zero_padding_' + str(block_num) + '_' + str(i + 1))(x)
        x = conv_layers[i](x)
        x = bn_layers[i](x, training=training)
        x = Activation('relu', name='relu_' + str(block_num) + '_' + str(i + 1))(x)
    return x


def get_correlation_model(num_spins, num_reps=3, training=True, embedding_length=DEFAULT_EMBEDDING_LENGTH, name='model'):
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
    inputs = Input(shape=(num_spins * num_reps, 3), name='inputs')
    x = inputs
    x = ZeroPadding1D(padding=tuple([int((ENCODER_KERNEL_SIZE - 1) / 2)] * 2), name='zero_padding_0')(x)
    x = Conv1D(CONV_BLOCK_FILTERS[0], ENCODER_KERNEL_SIZE, name='conv_0')(x)
    conv_layers, bn_layers = [], []
    for i in range(len(CONV_BLOCK_FILTERS)):
        conv_layers.append(Conv1D(filters=CONV_BLOCK_FILTERS[i],
                                  kernel_size=CONV_BLOCK_KERNELS_SIZES[i],
                                  strides=CONV_BLOCK_STRIDES[i],
                                  padding='valid',
                                  use_bias=False,
                                  kernel_regularizer=l2(REG),
                                  bias_regularizer=l2(REG),
                                  name='conv_' + str(i + 1))
                           )
        bn_layers.append(BatchNormalization(name='batchnorm_' + str(i + 1)))
    num_conv_blocks = int(np.log2(num_spins / embedding_length))
    for i in range(num_conv_blocks):
        x = apply_conv_block(x, conv_layers, bn_layers, block_num=i + 1, training=training)
        x = AveragePooling1D(pool_size=2, padding='valid', name='pooling_' + str(i + 1))(x)
    # x = Dense(units=1,
    #           kernel_regularizer=l2(REG),
    #           bias_regularizer=l2(REG),
    #           name='repeated_correlation')(x)
    # Switched to conv instead of dense because dense tends to output a single value for all samples.
    x = Conv1D(filters=1,
               kernel_size=1,
               kernel_regularizer=l2(REG),
               bias_regularizer=l2(REG),
               name='repeated_correlation')(x)
    x = Lambda(tf.squeeze, arguments={'axis': 2}, name='squeeze_1')(x)
    x = SliceRepetitions(final_size=DEFAULT_EMBEDDING_LENGTH, num_reps=num_reps)(x)
    # x = Lambda(tf.squeeze, arguments={'axis': 1}, name='squeeze_2')(x)
    # TODO - Consider intermediate dense layer.
    # x = Activation('sigmoid')(x)
    # x = Dense(1, activation='sigmoid')(x)
    # x = Lambda(lambda x: 2 * x - 1)(x)
    # x = Lambda(tf.squeeze, arguments={'axis': 1})(x)
    return Model(inputs=inputs, outputs=x, name=name)


def extend_model(orig_model, new_num_spins=None, num_reps=3, training=True, embedding_length=DEFAULT_EMBEDDING_LENGTH):
    orig_num_spins = int(orig_model.input.get_shape().as_list()[1] / num_reps)
    if new_num_spins is None:
        new_num_spins = orig_num_spins * 2
    new_model = get_correlation_model(num_spins=new_num_spins,
                                      num_reps=num_reps,
                                      training=training,
                                      embedding_length=embedding_length)
    orig_model.get_weights()






if __name__ == '__main__':
    model = get_correlation_model(num_spins=256, num_reps=3)
    model.summary()
    extended_model = extend_model(model)
