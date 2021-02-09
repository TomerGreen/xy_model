from data_generator import (
    RandomValueGenerator,
    CorrelationDataGenerator,
    Scaler,
)
from models import get_correlation_model, extend_model
from tensorflow.keras.optimizers import Optimizer, Adam, SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.python.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import json
from scipy.optimize import curve_fit
from scipy.stats import linregress


class SaveMetadata(Callback):
    def __init__(self, model, scaler=None, filepath=None):
        super(SaveMetadata, self).__init__()
        self.filepath = filepath
        self.metadata = {
            "input_size": model.input.get_shape().as_list()[1],
            "val_loss": 999999,
            "scaler_mean": scaler.mean,
            "scaler_std": scaler.std,
        }
        if scaler is not None:
            self.metadata["scaler_mean"] = scaler.mean
            self.metadata["scaler_std"] = scaler.std

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
    scaler=None,
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
        SaveMetadata(model, scaler, filepath=os.path.join(dirpath, "metadata.json")),
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


def mean_absolute_error(y, y_hat):
    assert len(y) == len(y_hat)
    y_hat = np.reshape(y_hat, newshape=y.shape)
    return np.mean(np.abs(y - y_hat) / (-y + 1e-10))


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


def present_data(x, y, pred=None, data_gen: CorrelationDataGenerator = None):
    dist = data_gen.get_distances_from_x(x)
    print(max(dist))
    inds = (dist >= np.max(dist) * (0)) & (dist <= np.max(dist) * (1))
    orig_slope, orig_intercept, orig_std_err = get_power_law(dist, y)
    lin_range = np.arange(min(dist[inds]), max(dist[inds]))
    orig_fit = np.exp(orig_slope * np.log(lin_range) + orig_intercept)
    plt.scatter(dist[inds], -y[inds], s=1.0, c="blue", label="pre-calculated")
    plt.plot(
        lin_range,
        orig_fit,
        label="analytical fit",
        c="black",
        linewidth=1,
        linestyle="dashed",
    )
    print("analytical slope:", orig_slope)
    if pred is not None:
        pred_slope, pred_intercept, pred_std_err = get_power_law(dist, pred)
        print("predicted_slope:", pred_slope)
        print("predicted_slope_error:", pred_std_err)
        print("intercept diff:", pred_intercept - orig_intercept)
        plt.scatter(dist[inds], -pred[inds], s=1.0, c="red", label="predicted")
        pred_fit = np.exp(pred_slope * np.log(lin_range) + pred_intercept)
        plt.plot(lin_range, pred_fit, label="predicted fit", c="black", linewidth=1)
    plt.suptitle("Pre-Calculated and Predicted Correlation by Spin Pair Distance")
    plt.xlabel("Distance (log scaled)")
    plt.ylabel("Absolute Correlation (log scaled)")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.show()


class XYModelSolver(object):
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler
        self.training_history = None

    def train(
        self,
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
        :return: A Keras history object
        """
        dirpath = os.path.join("trained_models", self.model.name)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
        self.model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
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
            SaveMetadata(
                self.model, self.scaler, filepath=os.path.join(dirpath, "metadata.json")
            ),
        ]
        sample_weight = None
        if weigh_long_interactions:
            sample_weight = 1 / (y + 1e-30)

        # Set Scaler
        self.scaler = Scaler(y)
        y = self.scaler.transform(y)

        # Train
        history = self.model.fit(
            x,
            y,
            callbacks=callbacks,
            epochs=epochs,
            validation_split=0.2,
            sample_weight=sample_weight,
        )
        # Save history
        self.training_history = history
        return history

    def predict(self, x):
        pass


def get_power_law(dist, corr):
    """
    Gets a vector of distances and a vector of correlations, and finds the power law based on linear
    fitting of the log-log plot.
    :param dist: Vector-like distances data in integers.
    :param corr: Corresponding correlation data in floats.
    :return:
    """
    num_spins = np.max(dist)
    low_lim = int(np.ceil(num_spins * (0)))
    high_lim = int(np.floor(num_spins * (1 / 10)))
    inds = (dist >= low_lim) & (dist <= high_lim)
    dist = dist[inds]
    corr = corr[inds]
    if len(corr.shape) == 2:
        corr = corr[:, 0]
    slope, intercept, r_value, p_value, std_err = linregress(
        np.log(dist), np.log(-corr)
    )
    return slope, intercept, std_err


def get_slopes(data_gen: CorrelationDataGenerator, model, scaler, max_length=32768):
    start_length = int(
        model.input.get_shape().as_list()[1] / 3
    )  # In case of 3 repetitions
    lengths = np.power(
        2, np.arange(int(np.log2(start_length)), int(np.log2(max_length)))
    )
    for length in lengths:
        extended_model = extend_model(model, length)
        x, y = data_gen.get_data(length, 10000, 10000, return_y=True)
        # plt.scatter(data_gen.get_distances_from_x(x), -y, s=0.5)
        # plt.yscale("log")
        # plt.xscale("log")
        # plt.show()
        dist = data_gen.get_distances_from_x(x)
        pred = extended_model.predict(x)
        pred = scaler.inverse_transform(pred) / (
            4 ** (np.log2(length) - int(np.log2(start_length)))
        )
        slope, intercept, std_err = get_power_law(dist, pred)
        present_data(x, y, pred=pred, data_gen=data_gen)
        print(f"length: {length}, slope: {slope}, err: {std_err}")


if __name__ == "__main__":
    J_val_gen = RandomValueGenerator("uniform", 1.0, 1.0)
    h_val_gen = RandomValueGenerator("uniform", 0.0, 0.0)
    data_gen = CorrelationDataGenerator(
        J_val_gen=J_val_gen,
        h_val_gen=h_val_gen,
        disorder=False,
        gaussian_smoothing_sigma=0,
        toy_model=False,
        two_index_channels=False,
        total_channels=3,
        envelope_only=True,
    )
    x, y = data_gen.get_data(1024, 100000, 100000)
    dist = data_gen.get_distances_from_x(x)
    x = x[dist < int(np.max(dist) / 4)]
    y = y[dist < int(np.max(dist) / 4)]
    plt.scatter(data_gen.get_distances_from_x(x), -y, s=0.5)
    plt.yscale("log")
    plt.xscale("log")
    plt.show()
    present_data(x, y, pred=None, data_gen=data_gen)
    scaler = Scaler(y)
    y = scaler.transform(y)
    model = get_correlation_model(
        num_spins=1024,
        num_reps=3,
        name="J1h0_1024",
        normalize=True,
        encoder=True,
        input_channels=3,
    )
    model.load_weights("./trained_models/J1h0_1024/weights.h5")
    # model = extend_model(model, 512)
    # train(model, x, y, epochs=250, scaler=scaler, optimizer=Adam(0.001))

    get_slopes(data_gen, model, scaler)

    # x1, y1 = data_gen.get_data(16, 10000, 10000)
    # dist1 = data_gen.get_distances_from_x(x1)
    # pred1 = model1.predict(x1)
    # pred1 = scaler.inverse_transform(pred1)
    # present_data(x1, y1, pred1, data_gen)
    # # slope1, intercept1, std_err1 = get_power_law(dist1, pred1)
    # # print(slope1, std_err1)
    #
    # model2 = extend_model(model1, 16)
    # x2, y2 = data_gen.get_data(16, 10000, 10000)
    # dist2 = data_gen.get_distances_from_x(x2)
    # pred2 = model2.predict(x2)
    # pred2 = scaler.inverse_transform(pred2)
    # present_data(x2, y2, pred2, data_gen)
    # slope2, std_err2 = get_power_law(dist2, pred2)
    # print(slope2, std_err2)
    #
    # model3 = extend_model(model1, 1024)
    # x3, y3 = data_gen.get_data(1024, 100000, 100000)
    # dist3 = data_gen.get_distances_from_x(x3)
    # pred3 = model3.predict(x3)
    # pred3 = scaler.inverse_transform(pred3) / 16
    # present_data(x3, y3, pred3, data_gen)
    # # slope3, std_err3 = get_power_law(dist3, pred3)
    # # print(slope3, std_err3)
    #
    # model4 = extend_model(model1, 2048)
    # x4, y4 = data_gen.get_data(2048, 100000, 100000)
    # dist4 = data_gen.get_distances_from_x(x4)
    # pred4 = model4.predict(x4)
    # pred4 = scaler.inverse_transform(pred4) / 64
    # present_data(x4, y4, pred4, data_gen)
    # # slope4, std_err4 = get_power_law(dist4, pred4)
    # # print(slope4, std_err4)
    #
    # model5 = extend_model(model1, 4096)
    # x5, y5 = data_gen.get_data(4096, 100000, 100000)
    # dist5 = data_gen.get_distances_from_x(x5)
    # pred5 = model5.predict(x5)
    # pred5 = scaler.inverse_transform(pred5) / 256
    # present_data(x5, y5, pred5, data_gen)
    # # slope5, std_err5 = get_power_law(dist5, pred5)
    # # print(slope5, std_err5)

    #
    # x3, y3 = data_gen.get_data(1024, 10000, 500)
    # model3 = extend_model(model1, 1024)
    # pred3 = model3.predict(x3)
    # pred3 = scaler.inverse_transform(pred3) / 16
    # present_data(x3, y3, pred3, data_gen)
    # err1024 = mean_absolute_error(y3, pred3)
    # print(err1024)
    # errs.append(err1024)
    #
    # x4, y4 = data_gen.get_data(2048, 10000, 500)
    # model4 = extend_model(model1, 2048)
    # pred4 = model4.predict(x4)
    # pred4 = scaler.inverse_transform(pred4) / 64
    # present_data(x4, y4, pred4, data_gen)
    # err2048 = mean_absolute_error(y4, pred4)
    # print(err2048)
    # errs.append(err2048)
    #
    # x5, y5 = data_gen.get_data(4096, 10000, 1000)
    # model5 = extend_model(model1, 4096)
    # pred5 = model5.predict(x5)
    # pred5 = scaler.inverse_transform(pred5) / 256
    # present_data(x5, y5, pred5, data_gen)
    # err4096 = mean_absolute_error(y5, pred5)
    # print(err4096)
    # errs.append(err4096)
    #
    # plt.plot(
    #     [256, 512, 1024, 2048, 4096],
    #     errs,
    #     "bo-",
    # )
    # plt.suptitle("Prediction Mean Absolute Percentage Error")
    # plt.xlabel("# Spins")
    # plt.ylabel("Prediction Error Rate")
    # plt.show()

    # J_val_gen = RandomValueGenerator("uniform", 0.5, 1.5)
    # h_val_gen = RandomValueGenerator("uniform", 1.0, 1.0)
    # data_gen = CorrelationDataGenerator(
    #     J_val_gen=J_val_gen,
    #     h_val_gen=h_val_gen,
    #     disorder=False,
    #     gaussian_smoothing_sigma=3,
    #     toy_model=False,
    # )
    # x, y = data_gen.get_data(256, 10000, 50)
    # scaler = Scaler(y)
    # y = scaler.transform(y)
    # model = get_correlation_model(
    #     num_spins=256,
    #     num_reps=3,
    #     name="ferro",
    #     normalize=True,
    #     norm_groups=32,
    # )
    # train(model, x, y, epochs=250)

    # J_val_gen = RandomValueGenerator("uniform", 0.0, 1.0)
    # h_val_gen = RandomValueGenerator("uniform", 0.0, 1.0)
    # data_gen = CorrelationDataGenerator(
    #     J_val_gen=J_val_gen,
    #     h_val_gen=h_val_gen,
    #     disorder=False,
    #     gaussian_smoothing_sigma=3,
    #     toy_model=False,
    # )
    # x, y = data_gen.get_data(256, 10000, 50)
    # scaler = Scaler(y)
    # y = scaler.transform(y)
    # model = get_correlation_model(
    #     num_spins=256,
    #     num_reps=3,
    #     name="multi_phase",
    #     normalize=True,
    #     norm_groups=32,
    # )
    # train(model, x, y, epochs=250)
    #
    # J_val_gen = RandomValueGenerator("uniform", 1.0, 1.0)
    # h_val_gen = RandomValueGenerator("uniform", 1.0, 1.0)
    # data_gen = CorrelationDataGenerator(
    #     J_val_gen=J_val_gen,
    #     h_val_gen=h_val_gen,
    #     disorder=False,
    #     gaussian_smoothing_sigma=3,
    #     toy_model=False,
    # )
    # x, y = data_gen.get_data(256, 10000, 50)
    # scaler = Scaler(y)
    # y = scaler.transform(y)
    # model = get_correlation_model(
    #     num_spins=256,
    #     num_reps=3,
    #     name="const",
    #     normalize=True,
    #     norm_groups=32,
    # )
    # train(model, x, y, epochs=250)

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
