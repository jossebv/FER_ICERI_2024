from keras.models import Sequential
import keras


def create_cnn(num_classes, dropout_rate=0.1):
    model = Sequential()
    model.add(
        keras.layers.Conv2D(
            filters=64,
            kernel_size=(20, 2),
            padding="same",
            activation="relu",
            input_shape=(478, 2, 1),
            data_format="channels_last",
        )
    )
    model.add(keras.layers.MaxPool2D((5, 1), padding="same"))
    model.add(
        keras.layers.Conv2D(
            filters=32, kernel_size=(10, 2), padding="same", activation="relu"
        )
    )
    model.add(keras.layers.MaxPool2D((5, 1), padding="same"))
    model.add(
        keras.layers.Conv2D(
            filters=16, kernel_size=(5, 2), padding="same", activation="relu"
        )
    )
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    model.build()

    print("[MODEL SUMMARY]")
    model.summary()

    return model
