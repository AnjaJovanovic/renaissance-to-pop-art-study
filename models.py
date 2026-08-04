"""VGG-lite CNN (Keras)."""

from tensorflow import keras
from tensorflow.keras import layers

from dataset import IMAGE_SIZE, class_names


def build_vgglite(num_classes=None, input_size=IMAGE_SIZE):
    """Manji VGG: 32-32 | 64-64 | 128-128 | 256-256 + dense head."""
    if num_classes is None:
        num_classes = len(class_names())

    inputs = keras.Input(shape=(input_size, input_size, 3))
    x = inputs

    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="vgg_lite")
    return model


if __name__ == "__main__":
    model = build_vgglite()
    model.summary()
    print("parametara:", model.count_params())
