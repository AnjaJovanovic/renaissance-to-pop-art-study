"""Ucitavanje Pandora18K splitova i batch generatori."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
SPLITS_DIR = ROOT / "data" / "splits"

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42


def load_splits(splits_dir=None):
    splits_dir = Path(splits_dir) if splits_dir else SPLITS_DIR
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")
    return train_df, val_df, test_df


def class_names(df=None):
    if df is None:
        df, _, _ = load_splits()
    return sorted(df["label"].unique())


def label_to_index(names=None):
    names = names or class_names()
    return {name: i for i, name in enumerate(names)}


def _resolve_path(filepath):
    p = Path(filepath)
    if not p.is_file():
        p = ROOT / filepath
    return p


def load_image(filepath, image_size=IMAGE_SIZE, augment=False):
    path = _resolve_path(filepath)
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)

    if augment:
        if np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # blagi random brightness
        factor = 0.85 + 0.3 * np.random.rand()
        arr = np.asarray(img).astype(np.float32) * factor
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    x = np.asarray(img, dtype=np.float32) / 255.0
    return x


class ImageDataset:
    """Jednostavan batch generator iz CSV-a (radi i bez TF)."""

    def __init__(
        self,
        dataframe,
        class_to_idx=None,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment=False,
        seed=SEED,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.class_to_idx = class_to_idx or label_to_index(class_names(self.df))
        self.num_classes = len(self.class_to_idx)
        self.indices = np.arange(len(self.df))

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_start(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __iter__(self):
        self.on_epoch_start()
        for start in range(0, len(self.df), self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            xs, ys = [], []
            for i in batch_idx:
                row = self.df.iloc[i]
                xs.append(load_image(row["filepath"], self.image_size, augment=self.augment))
                ys.append(self.class_to_idx[row["label"]])
            x = np.stack(xs, axis=0)
            y = np.eye(self.num_classes, dtype=np.float32)[ys]
            yield x, y


def make_datasets(
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    splits_dir=None,
):
    train_df, val_df, test_df = load_splits(splits_dir)
    names = class_names(train_df)
    mapping = label_to_index(names)

    train_ds = ImageDataset(
        train_df,
        class_to_idx=mapping,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        augment=True,
    )
    val_ds = ImageDataset(
        val_df,
        class_to_idx=mapping,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )
    test_ds = ImageDataset(
        test_df,
        class_to_idx=mapping,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
    )
    return train_ds, val_ds, test_ds, names


if __name__ == "__main__":
    train_ds, val_ds, test_ds, names = make_datasets()
    print("klasa:", len(names))
    print("batcheva train/val/test:", len(train_ds), len(val_ds), len(test_ds))
    x, y = next(iter(train_ds))
    print("batch x:", x.shape, "y:", y.shape, "x range:", float(x.min()), float(x.max()))
