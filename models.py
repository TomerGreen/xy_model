import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Add,
    BatchNormalization,
    Activation,
    Dense,
    ZeroPadding1D,
    Lambda,
    AveragePooling1D,
)
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import os
from data_generator import CorrelationDataGenerator, RandomValueGenerator, Scaler
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD


ENCODER_KERNEL_SIZE = 1
DEFAULT_EMBEDDING_LENGTH = 1
CONV_BLOCK_FILTERS = [32, 32, 32]
CONV_BLOCK_KERNELS_SIZES = [3, 3, 3]
CONV_BLOCK_STRIDES = [1, 1, 1]
REG = 1e-8
# REG = 0


def SliceRepetitions(final_size, num_reps):
    """
    A layer that takes the middle tile in a sequence of identical tiles. Used to undo the tiling we induce to simulate
    cyclical boundary conditions.
    """

    def slice_repetitions(inputs):
        input_shape = tf.shape(inputs)
        output = tf.slice(
            inputs, [0, final_size * int(num_reps / 2)], [input_shape[0], final_size]
        )
        # Workaround for compute_output_shape not being called
        output = tf.reshape(output, compute_output_shape(input_shape))
        return output

    def compute_output_shape(input_shape):
        return input_shape[0], final_size

    return Lambda(slice_repetitions, name="slice_repetitions")


def apply_conv_block(
    x, conv_layers, bn_layers, block_num, training=True, use_batchnorm=True
):
    if use_batchnorm:
        assert len(conv_layers) == len(bn_layers)
    # TODO - consider ResNet.
    for i in range(len(conv_layers)):
        x = ZeroPadding1D(
            tuple([int((conv_layers[i].kernel_size[0] - 1) / 2)] * 2),
            name="zero_padding_" + str(block_num) + "_" + str(i + 1),
        )(x)
        x = conv_layers[i](x)
        if use_batchnorm:
            x = bn_layers[i](x, training=training)
        x = Activation("relu", name="relu_" + str(block_num) + "_" + str(i + 1))(x)
    return x


def get_recurrent_module(normalize=True):
    """
    Returns the recurrent module as a list of layers.
    """
    module = []
    for i in range(len(CONV_BLOCK_FILTERS)):
        module.append(
            ZeroPadding1D(
                tuple([int((CONV_BLOCK_KERNELS_SIZES[i] - 1) / 2)] * 2),
                name="zero_padding_" + str(i + 1),
            )
        )
        module.append(
            Conv1D(
                filters=CONV_BLOCK_FILTERS[i],
                kernel_size=CONV_BLOCK_KERNELS_SIZES[i],
                strides=CONV_BLOCK_STRIDES[i],
                padding="valid",
                use_bias=False,
                kernel_regularizer=l2(REG),
                bias_regularizer=l2(REG),
                name="conv_" + str(i + 1),
            )
        )
        if normalize:
            module.append(InstanceNormalization(name="instnorm_" + str(i + 1)))
        module.append(Activation("relu", name="relu_" + str(i + 1)))
    module.append(AveragePooling1D(pool_size=2, padding="valid", name="avg_pooling"))
    return module


def get_correlation_model(
    num_spins,
    num_reps=3,
    embedding_length=DEFAULT_EMBEDDING_LENGTH,
    name="model",
    normalize=True,
    encoder=True,
    input_channels=3,
):
    """
    Implements a convolutional model that finds z_spin correlations from XY model data. The function constructs a
    convolution block and applies it repeatedly on the input, pooling by half after each time, until the input is at
    a predetermined size. It then applies a decoder block to obtain the correlation as a scalar.
    :param num_spins: the number of spins in the spin chain
    :param num_reps: an odd integer. the number of times the input repeats itself, to imitate periodic boundary
    conditions.
    :param embedding_length: Length of input before the decoder.
    :param name: The name of the returned model.
    :param normalize: Whether to use normalization in the conv blocks.
    :return: a model whose input is of shape(batch_size, num_spins, 3) and output is of shape (batch_size,)
    """
    inputs = Input(shape=(num_spins * num_reps, input_channels), name="inputs")
    x = inputs
    if encoder:
        x = ZeroPadding1D(
            padding=tuple([int((ENCODER_KERNEL_SIZE - 1) / 2)] * 2),
            name="zero_padding_0",
        )(x)
        x = Conv1D(CONV_BLOCK_FILTERS[0], ENCODER_KERNEL_SIZE, name="conv_0")(x)
    recurrent_module = get_recurrent_module(
        normalize=normalize,
    )
    num_recurrences = int(np.log2(num_spins / embedding_length))
    for i in range(num_recurrences):
        for layer in recurrent_module:
            x = layer(x)
    x = Conv1D(filters=1, kernel_size=1, name="repeated_correlation")(x)
    x = Lambda(tf.squeeze, arguments={"axis": 2}, name="squeeze_1")(x)
    x = SliceRepetitions(final_size=DEFAULT_EMBEDDING_LENGTH, num_reps=num_reps)(x)
    # TODO - Consider intermediate dense layer.
    return Model(inputs=inputs, outputs=x, name=name)


# def get_correlation_model2(
#     num_spins,
#     num_reps=3,
#     embedding_length=DEFAULT_EMBEDDING_LENGTH,
#     name="model",
#     normalize=True,
#     training=True,
# ):
#     conv_layers, bn_layers = [], []
#     for i in range(len(CONV_BLOCK_FILTERS)):
#         conv_layers.append(
#             Conv1D(
#                 filters=CONV_BLOCK_FILTERS[i],
#                 kernel_size=CONV_BLOCK_KERNELS_SIZES[i],
#                 strides=CONV_BLOCK_STRIDES[i],
#                 padding="valid",
#                 use_bias=False,
#                 kernel_regularizer=l2(REG),
#                 bias_regularizer=l2(REG),
#                 name="conv_" + str(i + 1),
#             )
#         )
#         if normalize:
#             bn_layers.append(BatchNormalization(name="batchnorm_" + str(i + 1)))
#     inputs = Input(shape=(num_spins * num_reps, input_channels), name="inputs")
#     x = inputs
#     if encoder:
#         x = ZeroPadding1D(
#             padding=tuple([int((ENCODER_KERNEL_SIZE - 1) / 2)] * 2),
#             name="zero_padding_0",
#         )(x)
#         x = Conv1D(CONV_BLOCK_FILTERS[0], ENCODER_KERNEL_SIZE, name="conv_0")(x)
#     num_recurrences = int(np.log2(num_spins / embedding_length))
#     for i in range(num_recurrences):
#         x = apply_conv_block(
#             x,
#             conv_layers,
#             bn_layers,
#             block_num=i + 1,
#             training=training,
#             use_batchnorm=normalize,
#         )
#         x = AveragePooling1D(
#             pool_size=2, padding="valid", name="pooling_" + str(i + 1)
#         )(x)
#     x = Conv1D(filters=1, kernel_size=1, name="repeated_correlation")(x)
#     x = Lambda(tf.squeeze, arguments={"axis": 2}, name="squeeze_1")(x)
#     x = SliceRepetitions(final_size=DEFAULT_EMBEDDING_LENGTH, num_reps=num_reps)(x)
#     # TODO - Consider intermediate dense layer.
#     return Model(inputs=inputs, outputs=x, name=name)


def extend_model(
    orig_model,
    new_num_spins=None,
    num_reps=3,
    embedding_length=DEFAULT_EMBEDDING_LENGTH,
    full_rnn=False,
):
    input_shape = orig_model.input.get_shape().as_list()
    orig_num_spins = int(input_shape[1] / num_reps)
    if new_num_spins is None:
        new_num_spins = orig_num_spins * 2
    encoder = not full_rnn
    input_channels = input_shape[2]
    extended_model = get_correlation_model(
        num_spins=new_num_spins,
        num_reps=num_reps,
        embedding_length=embedding_length,
        name=orig_model.name + "_" + str(new_num_spins),
        encoder=encoder,
        input_channels=input_channels,
    )
    orig_model.save_weights("temp.h5")
    extended_model.load_weights("temp.h5", by_name=True)
    os.remove("temp.h5")

    # Option 1 - rescaling the correlation
    # extended_model.trainable = False
    # x = extended_model(extended_model.inputs)
    # x = Dense(units=1, name="scale")(x)
    # new_model = Model(inputs=extended_model.inputs, outputs=x)
    # return new_model

    # Option 2 - letting the embedding layer train
    for layer in extended_model.layers:
        if layer.name == "repeated_correlation":
            break
        layer.trainable = False
    return extended_model


def mse(y, y_hat):
    assert len(y) == len(y_hat)
    y_hat = np.reshape(y_hat, newshape=y.shape)
    return np.sum((y - y_hat) ** 2) / len(y)


if __name__ == "__main__":
    model = get_correlation_model(num_spins=1024, num_reps=3, name="smoothed_05_diff")
    model.load_weights("trained_models/smoothed_05_diff_adam_0510-1714/weights.h5")
    J_val_gen = RandomValueGenerator("uniform", 1.5, 2.5)
    h_val_gen = RandomValueGenerator("uniform", 0.5, 1.5)
    data_gen = CorrelationDataGenerator(
        num_spins=256,
        J_val_gen=J_val_gen,
        h_val_gen=h_val_gen,
        disorder=False,
        gaussian_smoothing_sigma=5,
        toy_model=False,
    )

    # extended_model = extend_model(model, new_num_spins=1024)
    # extended_model.summary()

    # Just for scaling
    # _, y_for_caling = get_correlation_data(
    #     256,
    #     10000,
    #     50,
    #     3,
    #     disorder=False,
    #     J_low=1.5,
    #     J_high=2.5,
    #     h_low=0.5,
    #     h_high=1.5,
    #     custom_dist=False,
    #     gaussian_smoothing_sigma=5,
    #     toy_model=False,
    # )
    # scaler = Scaler(y_for_caling, log=True, normalize=True)
    # print("scaler:")
    # print(scaler.mean)
    # print(scaler.std)
    # x, y = get_correlation_data(
    #     1024,
    #     10000,
    #     50,
    #     3,
    #     disorder=False,
    #     J_low=1.5,
    #     J_high=2.5,
    #     h_low=0.5,
    #     h_high=1.5,
    #     custom_dist=False,
    #     gaussian_smoothing_sigma=5,
    #     toy_model=False,
    # )
    # print("data:")
    # print(y.mean())
    # print(y.std())
    # y = scaler.transform(y)
    # extended_model.compile(optimizer=Adam(), loss="mse", metrics=["mse"])
    # extended_model.fit(
    #     x, y, epochs=100, validation_split=0.2,
    # )
    # test_x, test_y = get_correlation_data(
    #     512,
    #     10000,
    #     50,
    #     3,
    #     disorder=False,
    #     J_low=1.5,
    #     J_high=2.5,
    #     h_low=0.5,
    #     h_high=1.5,
    #     custom_dist=False,
    #     gaussian_smoothing_sigma=5,
    #     toy_model=False,
    # )
    # pred = extended_model.predict(test_x)
    # # pred = scaler.inverse_transform(pred)
    # plt.hist(pred, bins=100)
    # plt.show()
    # print(pred.mean())
    # print(pred.max())
    # print(pred.min())
    # test_y = scaler.transform(test_y)
    # plt.hist(test_y, bins=100)
    # plt.show()
    # print(test_y.mean())
    # print(test_y.max())
    # print(test_y.min())
    # print(mse(test_y, pred))
