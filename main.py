from data_generator import (
    RandomValueGenerator,
    CorrelationDataGenerator,
    Scaler,
    get_distances_from_x,
)
from models import get_correlation_model, extend_model
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
        color = np.random.rand(3,)
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
    model = get_correlation_model(num_spins=256, num_reps=3, name="smoothed_05_diff")
    model.load_weights("trained_models/smoothed_05_diff/weights.h5")
    J_val_gen = RandomValueGenerator("uniform", 1.5, 2.5)
    h_val_gen = RandomValueGenerator("uniform", 0.5, 1.5)
    data_gen = CorrelationDataGenerator(
        J_val_gen=J_val_gen,
        h_val_gen=h_val_gen,
        disorder=False,
        gaussian_smoothing_sigma=5,
        toy_model=False,
    )
    # x, y = data_gen.get_data(num_spins=256, samples=10000, samples_per_config=50)
    # scaler = Scaler(y)
    # y = scaler.transform(y)
    # model, history = train(model, x, y, epochs=250, optimizer=Adam())
    # extended_model = extend_model(model, new_num_spins=1024)
    # ext_x, ext_y = data_gen.get_data(
    #     num_spins=1024, samples=10000, samples_per_config=50
    # )
    # ext_y = scaler.transform(ext_y)
    # train(extended_model, ext_x, ext_y, epochs=100, optimizer=Adam())
    # test_x, test_y = data_gen.get_data(
    #     num_spins=1024, samples=1000, samples_per_config=50
    # )
    # pred = extended_model.predict(test_x)
    # pred = scaler.inverse_transform(pred)
    # plt.scatter(get_distances_from_x(test_x), np.log(-test_y))
    # plt.scatter(get_distances_from_x(test_x), np.log(-pred))
    # plt.show()

    model = get_correlation_model(
        num_spins=256, num_reps=3, training=False, name="smoothed_05_diff"
    )
    model.load_weights("trained_models/smoothed_05_diff/weights.h5")
    short_x, short_y = data_gen.get_data(256, 10000, 50)
    scaler = Scaler(short_y)
    short_y = scaler.transform(short_y)
    short_pred = model.predict(short_x)
    print(mse(short_y, short_pred))
    short_pred = scaler.inverse_transform(short_pred)
    short_y = scaler.inverse_transform(short_y)
    print(mse(short_y, short_pred))
    # plt.scatter(get_distances_from_x(short_x), short_y, s=0.5, c="blue")
    plt.scatter(get_distances_from_x(short_x), short_pred, s=0.5, c="red")
    plt.show()

    extended_model = extend_model(model, new_num_spins=512, training=False)
    long_x, long_y = data_gen.get_data(512, 10000, 50)
    long_y = scaler.transform(long_y)
    long_pred = extended_model.predict(long_x)
    print(mse(long_y, long_pred))
    long_pred = scaler.inverse_transform(long_pred)
    long_y = scaler.inverse_transform(long_y)
    print(mse(long_y, long_pred))
    # plt.scatter(get_distances_from_x(long_x), np.log(-long_y), s=0.5, c="blue")
    plt.scatter(get_distances_from_x(long_x), np.log(-long_pred), s=0.5, c="red")
    plt.show()

    # x_distances = get_distances_from_x(test_x)
    # log_y = np.log(-test_y + 1e-30)
    # x_distances = x_distances
    # popt, pcov = curve_fit(log_func, x_distances, log_y)
    # a, b = popt
    # plt.scatter(np.log(x_distances), log_y, s=0.5, c="red")
    # # curve_x = np.arange(0, 512)
    # # plt.plot(curve_x, a * np.log(curve_x) + b)
    # plt.show()

    # # x, y = get_correlation_data(256, 100000, 500, 3, J_std=1.0, h_std=0.25, disorder=False, custom_dist=False)
    # x, y = get_correlation_data(
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
    # scaler = Scaler(y, log=True, normalize=False)
    # # scaler.fit(y)
    # y = scaler.transform(y)
    # plt.hist(y, bins=100)
    # plt.show()
    #
    # model = get_correlation_model(
    #     256,
    #     3,
    #     embedding_length=1,
    #     name="smoothed_05_diff_adam",
    #     training=True,
    #     use_batchnorm=True,
    # )
    # model.summary()
    # # model.load_weights('trained_models/smoothed_05_diff_adam_2809-2301/smoothed_05_diff_adam_2809-2301.h5')
    # model, history = train(
    #     model, x, y, epochs=10, weigh_long_interactions=False, optimizer=Adam()
    # )
    # plot_history(history)
    # # model = load_model('trained_models/smoothed_05_diff_adam_2809-2301/smoothed_05_diff_adam_2809-2301.h5')
    # test_x, test_y = get_correlation_data(
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
    # print("scaler:")
    # print(scaler.mean)
    # print(scaler.std)
    # pred = model.predict(test_x)
    # plt.hist(pred, bins=100)
    # plt.show()
    # print("pred:")
    # print(pred.mean())
    # print(pred.max())
    # print(pred.min())
    # test_y = scaler.transform(test_y)
    # plt.hist(test_y, bins=100)
    # plt.show()
    # print("test:")
    # print(test_y.mean())
    # print(test_y.max())
    # print(test_y.min())
    # print("test mse:")
    # print(mse(test_y, pred))
    #
    # # model_path = 'trained_models/model_1008-1643/model_1008-1643.h5'
    # # model = load_model(model_path, custom_objects={'SliceRepetitions': SliceRepetitions})
    # # model = get_correlation_model(num_spins=512, num_reps=3, training=True)
    # # model.load_weights('./trained_models/model_1008-1643/weights_model_1008-1643.h5',
    # #                    by_name=True)
    # # print("Loaded!")
    # # model.summary(150)
    # # test_x, test_y = get_correlation_data(512, 1000, 100, 3, disorder=False)
    # # pred = model.predict(test_x)
    # # print(test_y.shape)
    # # print(pred.shape)
    # # print(mse(test_y, pred))
    # # train(model, test_x, test_y, epochs=3)
    #
    # # model = get_correlation_model(num_spins=256, num_reps=3)
    # # model.summary(100)
    # # model.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
    # # train(model, x, y, epochs=250)
    # # test_x, test_y = get_correlation_data(256, 1000, 10, 3)
    # # print(test_y.shape)
    # # print(model.predict(test_x).shape)
    # # print(model.predict(test_x) - test_y)
