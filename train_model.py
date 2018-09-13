import os
import numpy as np

from keras import backend
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model

from data_management import normalize_data, load_data
from deep_predictive_coding.prednet_model_builder import PredNetModelBuilder
from deep_predictive_coding.model_type import ModelType
from settings import TESTS_CONFIG


def train(config):
    # Load training data
    print("Loading and normalizing data...")
    training_data = os.path.join(config.data_dir, "train")
    data_type = np.uint8 if config.model_type != ModelType.STATE_VECTOR else np.float32
    data = load_data(training_data, num_of_samples=config.sequences, dtype=data_type)
    data = normalize_data(data, config.model_type)
    print("Finished loading data!!")

    input_shape = data.shape[2:]

    # Initialize and train model
    print("Initilalizing model...")
    factory = PredNetModelBuilder(config)
    factory.set_input_shape(input_shape)
    factory.set_tensorboard_verbosity(write_grads=True, write_images=True)
    #factory.add_pretrained_autoencoder("C:\\Users\\Alberto\\Projects\\Keras_models\\autoencoder_bb\\autoencoder_model.json",
    #                                   "C:\\Users\\Alberto\\Projects\\Keras_models\\autoencoder_bb\\autoencoder_weights.hdf5")
    factory.trainable_autoencoder = False
    model, callbacks = factory.build_model(config.model_type)
    print("Done!")

    print("Training model...")
    y = np.zeros((data.shape[0], 1), np.float32)
    model.fit(data, y, config.batch_size, config.epochs, verbose=2, callbacks=callbacks,  validation_split=0.1)
    print("Done!")

    print("Saving model...")
    json_string = model.to_json()
    if not os.path.isdir(config.model_dir):
        os.mkdir(config.model_dir)
    json_file = os.path.join(config.model_dir, "prednet_model.json")
    with open(json_file, "w") as f:
        f.write(json_string)
    print("Done!")


def train_autoencoder(trainingData, outDir):
    # Load training data
    print("Loading and normalizing data...")
    data = load_data(trainingData)
    data = np.reshape(data, (data.shape[0],) + data.shape[2:])
    data = data.astype(np.float32) / 255
    print("Finished loading data!!")

    # Initialize and train model
    print("Initlializing model...")
    #data_format = backend.image_data_format()
    input = Input(shape=(48,64,1))
    autoencoder_layers = Conv2D(3, 3, padding='same', activation='relu')(input)
    autoencoder_layers = MaxPooling2D()(autoencoder_layers)
    autoencoder_layers = Conv2D(5, 3, padding='same', activation='relu')(autoencoder_layers)
    autoencoder_layers = MaxPooling2D()(autoencoder_layers)
    autoencoder_layers = UpSampling2D()(autoencoder_layers)
    autoencoder_layers = Conv2D(3, 3, padding='same', activation='relu')(autoencoder_layers)
    autoencoder_layers = UpSampling2D()(autoencoder_layers)
    autoencoder_layers = Conv2D(1, 3, padding='same', activation='sigmoid')(autoencoder_layers)
    autoencoder = Model(inputs=input, outputs=autoencoder_layers)
    autoencoder.compile(optimizer='sgd', loss='binary_crossentropy')
    print("Done!")

    batch_size = 256
    nb_epoch = 100

    print("Training model...")
    weights_file = os.path.join(outDir, "autoencoder_weights.hdf5")
    json_file    = os.path.join(outDir, "autoencoder_model.json")
    tbLogPath    = os.path.join(outDir, "tensorboard_logs")
    callbacks = [ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True),
                 TensorBoard(log_dir=tbLogPath,histogram_freq=nb_epoch/10, batch_size=batch_size)]
    autoencoder.fit(data, data, batch_size, nb_epoch, verbose=2, callbacks=callbacks, validation_split=0.1)
    json_string = autoencoder.to_json()
    with open(json_file, "w") as f:
        f.write(json_string)
    print("Done!")


if __name__ == "__main__":
    #train_autoencoder("C:\\Users\\Alberto\\Projects\\Datasets\\bouncing_balls\\autoencoder_dataset\\train",
    #                  "C:\\Users\\Alberto\\Projects\\Keras_models\\autoencoder_bb")

    for config in TESTS_CONFIG:
        print("Training dataset ", str(config))
        print("\n\nNUMBER OF EPOCHS: ", config.epochs)
        train(config)
        backend.clear_session()