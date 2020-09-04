from data_generator import get_correlation_data, Scaler
from models import get_correlation_model
from layers import SliceRepetitions
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import json


class SaveMetadata(Callback):

    def __init__(self, model, filepath):
        super(SaveMetadata, self).__init__()
        self.filepath = filepath
        self.metadata = {
            'input_size': model.input.get_shape().as_list()[1],
            'val_loss': 999999
        }

    def write_json(self):
        with open(self.filepath, 'w') as file:
            file.write(json.dumps(self.metadata, file, indent=4))

    def on_train_begin(self, logs=None):
        self.write_json()

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.metadata['val_loss']:
            self.metadata['val_loss'] = logs['val_loss']
            self.write_json()



def train(model, x, y, epochs=10, optimizer=Adam(), long_interactions_wieght=False):
    timestamp = datetime.now().strftime("%d%m-%H%M")
    dirname = model.name + '_' + timestamp
    dirpath = os.path.join('trained_models', dirname)
    os.mkdir(dirpath)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    callbacks = [
        # Saving both the model and its weights.
        ModelCheckpoint(os.path.join(dirpath, dirname + '.h5'),
                        verbose=2,
                        monitor='val_loss',
                        save_best_only=True),
        ModelCheckpoint(os.path.join(dirpath, dirname + '_weights.h5'),
                        verbose=2,
                        monitor='val_loss',
                        save_best_only=True,
                        save_weights_only=True),
        # For some reason SaveMetadata.model returns None
        SaveMetadata(model, filepath=os.path.join(dirpath, 'metadata.json'))
    ]
    sample_weight = None
    if long_interactions_wieght:
        sample_weight = 1/(y + 1e-20)
    model.fit(x,
              y,
              callbacks=callbacks,
              epochs=epochs,
              validation_split=0.2,
              sample_weight=sample_weight
              )
    return model


def compare_models(models, x, y, epochs):
    histories = []
    for model in models:
        history = train(model, x, y, epochs=epochs)
        histories.append(history)
    for i, model in enumerate(models):
        color = np.random.rand(3,)
        plt.plot(range(epochs), histories[i].history['loss'], c=color, label=model.name)
        plt.plot(range(epochs), histories[i].history['val_loss'], ls='--', c=color)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def mse(y, y_hat):
    assert len(y) == len(y_hat)
    y_hat = np.reshape(y_hat, newshape=y.shape)
    return np.sum((y - y_hat) ** 2) / len(y)


def check_same_weights(model1, model2):
    pass


if __name__ == '__main__':
    x, y = get_correlation_data(256, 10000, 50, 3, disorder=False, custom_dist=False)
    scaler = Scaler()
    scaler.fit(y)
    y = scaler.transform(y)
    model = get_correlation_model(256, 3, embedding_length=1, name='long_interactions_log_model', training=True)
    model.summary()
    model = train(model, x, y, epochs=2, long_interactions_wieght=False)
    # model.load_weights('trained_models/long_interactions_norm_model_1808-1009/long_interactions_norm_model_1808-1009_weights.h5')
    # from tensorflow.python.keras.models import Model
    # test_model = Model(inputs=model.inputs, outputs=model.get_layer('conv_0').output)
    test_x, test_y = get_correlation_data(256, 10000, 50, 3, disorder=False, custom_dist=False)
    # scaler = Scaler(test_y)
    # test_y = scaler.transform(test_y)
    pred = model.predict(test_x)
    pred = scaler.inverse_transform(pred)
    print(test_y.mean())
    print(test_y.std())
    print(pred.mean())
    print(pred.std())
    print(mse(test_y, pred))
    plt.hist(test_y)
    plt.show()
    plt.hist(pred)
    plt.show()

    # model_path = 'trained_models/model_1008-1643/model_1008-1643.h5'
    # model = load_model(model_path, custom_objects={'SliceRepetitions': SliceRepetitions})
    # model = get_correlation_model(num_spins=512, num_reps=3, training=True)
    # model.load_weights('./trained_models/model_1008-1643/weights_model_1008-1643.h5',
    #                    by_name=True)
    # print("Loaded!")
    # model.summary(150)
    # test_x, test_y = get_correlation_data(512, 1000, 100, 3, disorder=False)
    # pred = model.predict(test_x)
    # print(test_y.shape)
    # print(pred.shape)
    # print(mse(test_y, pred))
    # train(model, test_x, test_y, epochs=3)


    # model = get_correlation_model(num_spins=256, num_reps=3)
    # model.summary(100)
    # model.compile(optimizer=Adam(), loss='mse', metrics=['mse'])
    # train(model, x, y, epochs=250)
    # test_x, test_y = get_correlation_data(256, 1000, 10, 3)
    # print(test_y.shape)
    # print(model.predict(test_x).shape)
    # print(model.predict(test_x) - test_y)