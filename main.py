from data_generator import (
    RandomValueGenerator,
    CorrelationDataGenerator,
    Scaler,
    get_distances_from_x,
)
from models import get_correlation_model, get_correlation_model2, extend_model
from tensorflow.keras.optimizers import Optimizer, Adam, SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.python.keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import json
from scipy.optimize import curve_fit


class SaveMetadata(Callback):
    def __init__(self, model, filepath):
        super(SaveMetadata, self).__init__()
        self.filepath = filepath
        self.metadata = {
            "input_size": model.input.get_shape().as_list()[1],
            "val_loss": 999999,
        }

    def write_json(self):
        with open(self.filepath, "w") as file:
            file.write(json.dumps(self.metadata, indent=4))

    def on_train_begin(self, logs=None):
        self.write_json()

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] < self.metadata["val_loss"]:
            self.metadata["val_loss"] = logs["val_loss"]
            self.write_json()


def train(
    model,
    x,
    y,
    epochs: int = 10,
    optimizer: Optimizer = Adam(),
    weigh_long_interactions=False,
):
    """
    Trains the model using scaled data.
    :param epochs: Number of epochs.
    :param optimizer: Optimizer used.
    :param weigh_long_interactions: If True, we use a predefined function to give more
    training weight to longer-interacting samples.
    :return: A Keras Model object and a Keras history object, as a tuple.
    """
    dirpath = os.path.join("trained_models", model.name)
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
    callbacks = [
        # Saving both the model and its weights.
        ModelCheckpoint(
            os.path.join(dirpath, "model.h5"),
            verbose=2,
            monitor="val_loss",
            save_best_only=True,
        ),
        ModelCheckpoint(
            os.path.join(dirpath, "weights.h5"),
            verbose=2,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
        # For some reason SaveMetadata.model returns None
        SaveMetadata(model, filepath=os.path.join(dirpath, "metadata.json")),
    ]
    sample_weight = None
    if weigh_long_interactions:
        sample_weight = 1 / (y + 1e-30)
    history = model.fit(
        x,
        y,
        callbacks=callbacks,
        epochs=epochs,
        validation_split=0.2,
        sample_weight=sample_weight,
    )
    return model, history


def compare_models(models, x, y, epochs):
    histories = []
    for model in models:
        history = train(model, x, y, epochs=epochs)
        histories.append(history)
    for i, model in enumerate(models):
        color = np.random.rand(
            3,
        )
        plt.plot(range(epochs), histories[i].history["loss"], c=color, label=model.name)
        plt.plot(range(epochs), histories[i].history["val_loss"], ls="--", c=color)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def mse(y, y_hat):
    assert len(y) == len(y_hat)
    y_hat = np.reshape(y_hat, newshape=y.shape)
    return np.sum((y - y_hat) ** 2) / len(y)


def check_same_weights(model1, model2):
    pass


def plot_history(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def log_func(x, a, b):
    return a * np.log(x) + b


if __name__ == "__main__":
    model = get_correlation_model(
        num_spins=256, num_reps=3, name="layernorm_2_to_1", normalize=True
    )
    model.summary(150)
    J_val_gen = RandomValueGenerator("uniform", 2.0, 2.0)
    h_val_gen = RandomValueGenerator("uniform", 1.0, 1.0)
    data_gen = CorrelationDataGenerator(
        J_val_gen=J_val_gen,
        h_val_gen=h_val_gen,
        disorder=False,
        gaussian_smoothing_sigma=5,
        toy_model=False,
    )
    x, y = data_gen.get_data(256, 1000, 50)
    scaler = Scaler(y)
    y = scaler.transform(y)
    train(model, x, y, epochs=10)
    test_x, test_y = data_gen.get_data(256, 1000, 50)
    pred = model.predict(test_x)
    print(pred.min())
    print(pred.max())
    pred = scaler.inverse_transform(pred)
    plt.scatter(np.log(get_distances_from_x(test_x)), np.log(-test_y), s=0.5, c="blue")
    plt.scatter(np.log(get_distances_from_x(test_x)), np.log(-pred), s=0.5, c="red")
    plt.show()

    # model.summary(150)
    # model1 = get_correlation_model(num_spins=256, num_reps=3, name="smoothed_05_diff", training=True)
    # model1.load_weights("trained_models/smoothed_05_diff/weights.h5")
    # new_model11 = Model(inputs=model1.inputs, outputs=model1.get_layer("zero_padding_1_3").output)
    # new_pred11 = new_model11.predict(short_x)
    # print(new_pred11[0].shape)
    # print(new_pred11[0][:, :].mean())
    # print(new_pred11[0][:, :].var())
    # new_model12 = Model(inputs=model1.inputs, outputs=model1.get_layer("zero_padding_5_3").output)
    # new_pred12 = new_model12.predict(short_x)
    # print(new_pred12[0].shape)
    # print(new_pred12[0][:, :].mean())
    # print(new_pred12[0][:, :].var())
    # new_model13 = Model(inputs=model1.inputs, outputs=model1.get_layer("zero_padding_8_3").output)
    # new_pred13 = new_model13.predict(short_x)
    # print(new_pred13[0].shape)
    # print(new_pred13[0][:, :].mean())
    # print(new_pred13[0][:, :].var())
    #
    # model2 = get_correlation_model(num_spins=256, num_reps=3, name="smoothed_05_diff", training=False)
    # model2.load_weights("trained_models/smoothed_05_diff/weights.h5")
    # new_model11 = Model(inputs=model2.inputs, outputs=model2.get_layer("zero_padding_1_3").output)
    # new_pred11 = new_model11.predict(short_x)
    # print(new_pred11[0].shape)
    # print(new_pred11[0][:, :].mean())
    # print(new_pred11[0][:, :].var())
    # new_model12 = Model(inputs=model2.inputs, outputs=model2.get_layer("zero_padding_5_3").output)
    # new_pred12 = new_model12.predict(short_x)
    # print(new_pred12[0].shape)
    # print(new_pred12[0][:, :].mean())
    # print(new_pred12[0][:, :].var())
    # new_model13 = Model(inputs=model2.inputs, outputs=model2.get_layer("zero_padding_8_3").output)
    # new_pred13 = new_model13.predict(short_x)
    # print(new_pred13[0].shape)
    # print(new_pred13[0][:, :].mean())
    # print(new_pred13[0][:, :].var())
    #
    # print(mse(short_y, short_pred))
    # short_pred = scaler.inverse_transform(short_pred)
    # short_y = scaler.inverse_transform(short_y)
    # print(mse(short_y, short_pred))
    # plt.scatter(get_distances_from_x(short_x), np.log(-short_y), s=0.5, c="blue")
    # plt.scatter(get_distances_from_x(short_x), np.log(-short_pred), s=0.5, c="red")
    # plt.show()

    # extended_model = extend_model(model, new_num_spins=512, training=True)
    # extended_model.summary(150)
    # long_x, long_y = data_gen.get_data(512, 10000, 50)
    # long_y = scaler.transform(long_y)
    # long_pred = extended_model.predict(long_x, batch_size=1)
    # print(mse(long_y, long_pred))
    # long_pred = scaler.inverse_transform(long_pred)
    # long_y = scaler.inverse_transform(long_y)
    # print(mse(long_y, long_pred))
    # plt.scatter(np.log(get_distances_from_x(long_x)), np.log(-long_y), s=0.5, c="blue")
    # plt.scatter(np.log(get_distances_from_x(long_x)), np.log(-long_pred), s=0.5, c="red")
    # plt.show()
