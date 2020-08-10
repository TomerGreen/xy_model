from data_generator import get_correlation_data
from models import get_correlation_model
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import json


class SaveMetadata(Callback):

    def __init__(self, filepath):
        super(SaveMetadata, self).__init__()
        self.filepath = filepath
        self.metadata = {
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



def train(model, x, y, epochs=10, optimizer=Adam()):
    timestamp = datetime.now().strftime("%d%m-%H%M")
    dirname = model.name + '_' + timestamp
    dirpath = os.path.join('trained_models', dirname)
    os.mkdir(dirpath)
    model.compile(optimizer=optimizer, loss='mse')
    callbacks = [
        ModelCheckpoint(os.path.join(dirpath, dirname + '.h5'),
                        verbose=2,
                        monitor='val_loss',
                        save_best_only=True),
        SaveMetadata(filepath=os.path.join(dirpath, 'metadata.json'))
    ]
    history = model.fit(x, y, callbacks=callbacks, epochs=epochs, validation_split=0.2)
    return history


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


if __name__ == '__main__':
    # x, y = get_correlation_data(256, 100000, 50, 3, disorder=False)
    # model1 = get_correlation_model(256, 3, embedding_length=1)
    # model1.summary()
    # compare_models([model1], x, y, 10)

    model_path = 'trained_models/model_2906-2013/2906-2013.h5'
    model = load_model(model_path)
    test_data = get_correlation_data(256, 1, 1, 3, disorder=False)
    pred = model.predict(test_data)
    print(test_data)
    print(pred)


    # model = get_correlation_model(256, 3)
    # model.summary(100)
    # model.compile(optimizer=Adam(), loss='mse')
    # model.fit(x, y, batch_size=64, epochs=10, validation_split=0.2)
    # model.save('./test_model.h5')
    # test_x, test_y = get_correlation_data(256, 100, 10, 3)
    # print(test_y.shape)
    # print(model.predict(test_x).shape)
    # print(model.predict(test_x) - test_y)