# VGG-lite i hybrid CNN

from tensorflow import keras
from tensorflow.keras import layers

from dataset import IMAGE_SIZE, class_names


def build_vgglite(num_classes=None, input_size=IMAGE_SIZE):
    # manji VGG: 4 bloka (2x Conv + MaxPool), filteri 32/64/128/256, pa dense + softmax
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


def _inception_block(x):
    # 4 grane: 1x1, 1x1->3x3, 1x1->5x5, maxpool->1x1, pa concat
    b1 = layers.Conv2D(32, 1, padding="same", activation="relu")(x)

    b3 = layers.Conv2D(16, 1, padding="same", activation="relu")(x)
    b3 = layers.Conv2D(64, 3, padding="same", activation="relu")(b3)

    b5 = layers.Conv2D(8, 1, padding="same", activation="relu")(x)
    b5 = layers.Conv2D(32, 5, padding="same", activation="relu")(b5)

    bp = layers.MaxPooling2D(3, strides=1, padding="same")(x)
    bp = layers.Conv2D(32, 1, padding="same", activation="relu")(bp)

    return layers.Concatenate()([b1, b3, b5, bp])


def build_hybrid(num_classes=None, input_size=IMAGE_SIZE):
    # tankiji VGG stem (8/16/32/64) + jedna inception grana + mali dense
    if num_classes is None:
        num_classes = len(class_names())

    inputs = keras.Input(shape=(input_size, input_size, 3))
    x = inputs

    for filters in (8, 16, 32, 64):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)

    x = _inception_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="hybrid")
    return model


if __name__ == "__main__":
    for name, builder in (("vgglite", build_vgglite), ("hybrid", build_hybrid)):
        model = builder()
        model.summary()
        print(name, "parametara:", model.count_params())
        print("---")
