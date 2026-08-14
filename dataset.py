from pathlib import Path

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parent
SPLITS_DIR = ROOT / "data" / "splits"

# ulazna velicina slike za mrezu
IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42


def load_splits(splits_dir=None):
    # ucitava fiksirane train/val/test CSV liste
    splits_dir = Path(splits_dir) if splits_dir else SPLITS_DIR
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    return train_df, val_df, test_df


def class_names(df=None):
    # sortirane labele klasa (isti je redosled u svim generatorima)
    if df is None:
        df, _, _ = load_splits()
    return sorted(df["label"].unique())


def _with_abs_paths(df):
    # CSV cuva relativne putanje
    out = df.copy()
    out["filepath"] = out["filepath"].apply(
        lambda p: str(p if Path(p).is_file() else ROOT / p)
    )
    return out

def make_generators(
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    splits_dir=None,
):
    # train/val/test generatori
    # augmentacija samo na train
    train_df, val_df, test_df = load_splits(splits_dir)
    train_df = _with_abs_paths(train_df)
    val_df = _with_abs_paths(val_df)
    test_df = _with_abs_paths(test_df)
    names = class_names(train_df)

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
    )
    # bez augmentacije, da evaluacija bude stabilna
    plain_gen = ImageDataGenerator(rescale=1.0 / 255)

    common = dict(
        directory=None,
        x_col="filepath",
        y_col="label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        classes=names,
    )

    train_iter = train_gen.flow_from_dataframe(
        train_df, shuffle=True, seed=SEED, **common
    )
    val_iter = plain_gen.flow_from_dataframe(val_df, shuffle=False, **common)
    test_iter = plain_gen.flow_from_dataframe(test_df, shuffle=False, **common)
    return train_iter, val_iter, test_iter, names


def make_datasets(*args, **kwargs):
    return make_generators(*args, **kwargs)


if __name__ == "__main__":
    train_iter, val_iter, test_iter, names = make_generators()
    print("klasa:", len(names))
    x, y = next(train_iter)
    print("batch x:", x.shape, "y:", y.shape)
