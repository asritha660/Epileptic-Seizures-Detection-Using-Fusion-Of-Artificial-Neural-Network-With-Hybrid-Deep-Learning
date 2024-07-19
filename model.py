
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.callbacks import EarlyStopping

## deep learning model
def cnn_GRU_NN(X_train):
    model = keras.models.Sequential()

    model.add(layers.Conv2D(filters=64, kernel_size=(2, 4), padding='same', activation='relu', input_shape=X_train.shape[1:]))
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 4), strides=(1, 2),padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((1, 2)))

    model.add(layers.Conv2D(filters=128, kernel_size=(2, 4), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=128, kernel_size=(2, 4), strides=(1, 2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=256, kernel_size=(4, 4), padding='same', activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=(4, 4), strides=(1, 2), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((1, 2)))

    model.add(layers.GlobalAveragePooling2D())
    #model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model)
    return model

