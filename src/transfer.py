# Transfer learning: InceptionV3 sa zamrznutom bazom i novom dense glavom.
#
# Baza se propusta kroz slike samo jednom, a 2048-dim izlazi se kesiraju na disk
# (trik iz rada) — posle toga se glava trenira za sekunde po epohi.
#
# Druga faza je fine-tuning: poslednji Inception blok se odmrzava i uci se zajedno sa
# glavom, sa mnogo manjim lr. Tu kes ne moze da se koristi (baza se menja), pa se ide
# kroz same slike.
#
# Pokretanje:
#   python src/transfer.py                    # kesiranje + trening glave
#   python src/transfer.py --stage cache      # samo kesiranje
#   python src/transfer.py --stage dense      # samo trening glave (koristi postojeci kes)
#   python src/transfer.py --stage finetune   # fine-tune poslednjeg bloka, init iz dense glave

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from dataset import SEED, class_names, load_splits, make_generators, _with_abs_paths

# repo root je jedan nivo iznad src/
ROOT = Path(__file__).resolve().parents[1]

# InceptionV3 je pretreniran na 299x299
IMAGE_SIZE = 299
FEATURE_DIM = 2048

# prvi sloj poslednjeg Inception modula — sve pre njega ostaje zamrznuto
UNFREEZE_FROM = "mixed9"


def parse_args():
    p = argparse.ArgumentParser(description="Transfer learning (InceptionV3) na Pandora18K")
    p.add_argument("--stage", choices=("cache", "dense", "finetune", "all"), default="all")
    p.add_argument("--exp-dir", default=None, help="podrazumevano experiments/transfer_<stage>")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--batch-size", type=int, default=16, help="batch pri kesiranju (inferencija)")
    p.add_argument("--head-batch-size", type=int, default=64, help="batch pri treningu glave")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--recompute", action="store_true", help="prepisi postojeci kes")
    p.add_argument("--limit", type=int, default=None, help="uzmi samo N slika po skupu (brza provera)")

    # fine-tuning ima svoje podrazumevane vrednosti — mnogo manji lr i manje epoha
    ft = p.add_argument_group("fine-tuning (--stage finetune)")
    ft.add_argument("--ft-epochs", type=int, default=30)
    ft.add_argument("--ft-lr", type=float, default=1e-5)
    ft.add_argument("--ft-batch-size", type=int, default=16)
    ft.add_argument("--unfreeze-from", default=UNFREEZE_FROM,
                    help="ime sloja u InceptionV3 od kog se odmrzava")
    ft.add_argument("--dense-dir", default=None,
                    help="folder dense eksperimenta iz kog se uzima head_best.keras")
    ft.add_argument("--patience", type=int, default=6, help="early stopping po val_accuracy")
    ft.add_argument("--steps", type=int, default=None, help="brza provera: koraka po epohi")
    ft.add_argument("--val-steps", type=int, default=None)
    return p.parse_args()


def build_feature_extractor(image_size=IMAGE_SIZE):
    # Rescaling je unutar modela: generator daje [0,1], InceptionV3 trazi [-1,1].
    # Tako predobrada putuje zajedno sa sacuvanim modelom i evaluate.py radi neizmenjen.
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = layers.Rescaling(2.0, offset=-1.0, name="inception_preprocess")(inputs)

    base = InceptionV3(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
    base.trainable = False

    x = base(x, training=False)
    outputs = layers.GlobalAveragePooling2D(name="gap")(x)
    return keras.Model(inputs, outputs, name="inception_features")


def build_head(num_classes, dropout=0.5, feature_dim=FEATURE_DIM):
    # glava po planu: Dropout -> Dense(softmax) nad GAP izlazom
    inputs = keras.Input(shape=(feature_dim,))
    x = layers.Dropout(dropout)(inputs)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="transfer_head")


def _rel_to_root(path):
    # eksperiment moze biti i van repoa, pa relative_to nije uvek moguc
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_finetune_model(num_classes, image_size=IMAGE_SIZE, dropout=0.5, unfreeze_from=UNFREEZE_FROM):
    # Ista topologija kao sklopljeni dense model (slika -> softmax), ali je baza ovde
    # deo grafa koji se uci, pa se glava ne moze drzati odvojeno.
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = layers.Rescaling(2.0, offset=-1.0, name="inception_preprocess")(inputs)

    base = InceptionV3(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
    base.trainable = True

    layer_names = [layer.name for layer in base.layers]
    if unfreeze_from not in layer_names:
        raise SystemExit(f"nema sloja '{unfreeze_from}' u InceptionV3")
    cut = layer_names.index(unfreeze_from)

    for layer in base.layers[:cut]:
        layer.trainable = False
    for layer in base.layers[cut:]:
        # BatchNorm ostaje zamrznut: sa malim batch-om bi pokvario ImageNet statistike
        layer.trainable = not isinstance(layer, layers.BatchNormalization)

    # training=False drzi ceo BatchNorm u inferencijskom rezimu, i u odmrznutom delu
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="head_dense")(x)

    model = keras.Model(inputs, outputs, name="transfer_finetune")
    print(f"odmrznuto od '{unfreeze_from}' ({len(base.layers) - cut} slojeva baze)")
    return model


def init_head_from_dense(model, head_path):
    # Fine-tune krece od vec naucene glave, ne od nasumicne — inace bi prvi gradijenti
    # kroz odmrznutu bazu bili sum i pokvarili pretrenirane tezine.
    head = keras.models.load_model(head_path)
    dense = [layer for layer in head.layers if isinstance(layer, layers.Dense)][-1]
    model.get_layer("head_dense").set_weights(dense.get_weights())
    print("glava inicijalizovana iz:", head_path)


def count_trainable(model):
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def finetune(exp_dir, args, dense_dir):
    head_path = dense_dir / "head_best.keras"
    if not head_path.is_file():
        raise SystemExit(f"nema naucene glave na putanji: {head_path} (pokreni prvo --stage dense)")

    # augmentacija samo na train, val bez nje — ista konvencija kao kod custom modela
    train_iter, val_iter, _, names = make_generators(
        image_size=args.image_size,
        batch_size=args.ft_batch_size,
    )

    keras.utils.set_random_seed(SEED)
    model = build_finetune_model(
        num_classes=len(names),
        image_size=args.image_size,
        dropout=args.dropout,
        unfreeze_from=args.unfreeze_from,
    )
    init_head_from_dense(model, head_path)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.ft_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("parametara ukupno:", model.count_params(), "| trenabilnih:", count_trainable(model))

    ckpt_path = exp_dir / "best.keras"
    callbacks = [
        ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        CSVLogger(str(exp_dir / "history.csv"), append=False),
        EarlyStopping(monitor="val_accuracy", patience=args.patience, verbose=1),
    ]

    fit_kwargs = dict(
        x=train_iter,
        validation_data=val_iter,
        epochs=args.ft_epochs,
        callbacks=callbacks,
    )
    if args.steps is not None:
        fit_kwargs["steps_per_epoch"] = args.steps
    if args.val_steps is not None:
        fit_kwargs["validation_steps"] = args.val_steps

    history = model.fit(**fit_kwargs)

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history.history, f, indent=2)
    model.save(exp_dir / "last.keras")

    meta = {
        "model": "transfer_finetune",
        "num_classes": len(names),
        "image_size": args.image_size,
        "batch_size": args.ft_batch_size,
        "epochs": args.ft_epochs,
        "lr": args.ft_lr,
        "dropout": args.dropout,
        "params": int(model.count_params()),
        "trainable_params": count_trainable(model),
        "cached_features": False,
        "augmentation": True,
        "unfreeze_from": args.unfreeze_from,
        "init_from": _rel_to_root(head_path),
        "steps": args.steps,
        "val_steps": args.val_steps,
    }
    with open(exp_dir / "run_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    best_val = max(history.history["val_accuracy"])
    print()
    print("najbolji val_accuracy:", round(float(best_val), 4))
    print("best model:", ckpt_path)
    print("run_config:", exp_dir / "run_config.json")


def _plain_iter(df, names, image_size, batch_size, limit=None):
    # bez augmentacije i bez shuffle-a — inace kes ne bi odgovarao labelama
    df = _with_abs_paths(df)
    if limit is not None:
        df = df.groupby("label", group_keys=False).head(max(1, limit // len(names)))

    gen = ImageDataGenerator(rescale=1.0 / 255)
    return gen.flow_from_dataframe(
        df,
        directory=None,
        x_col="filepath",
        y_col="label",
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        classes=names,
        shuffle=False,
    )


def cache_features(exp_dir, image_size, batch_size, limit=None, recompute=False):
    # racuna 2048-dim izlaze zamrznute baze i pise ih u features/<split>.npz
    feat_dir = exp_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    splits = dict(zip(("train", "val", "test"), load_splits()))
    names = class_names()

    targets = {
        split: feat_dir / f"{split}.npz"
        for split in splits
        if recompute or not (feat_dir / f"{split}.npz").is_file()
    }
    if not targets:
        print("kes vec postoji, preskacem (--recompute da se prepise)")
        return names

    extractor = build_feature_extractor(image_size)

    for split, out_path in targets.items():
        it = _plain_iter(splits[split], names, image_size, batch_size, limit)
        steps = math.ceil(it.samples / batch_size)
        print(f"[{split}] {it.samples} slika, {steps} koraka")

        features = extractor.predict(it, steps=steps, verbose=1)
        features = features[: it.samples]
        # flow_from_dataframe vraca classes kao listu, ne niz
        labels = np.asarray(it.classes)[: it.samples]

        assert len(features) == len(labels), f"{split}: {len(features)} feature-a vs {len(labels)} labela"
        np.savez_compressed(out_path, X=features.astype("float32"), y=labels.astype("int32"))
        print(f"[{split}] sacuvano: {out_path}  X={features.shape}")

    return names


def load_cached(exp_dir, split):
    data = np.load(exp_dir / "features" / f"{split}.npz")
    return data["X"], data["y"]


def train_head(exp_dir, args, names):
    num_classes = len(names)

    x_train, y_train = load_cached(exp_dir, "train")
    x_val, y_val = load_cached(exp_dir, "val")
    print("train:", x_train.shape, "val:", x_val.shape)

    y_train_oh = keras.utils.to_categorical(y_train, num_classes)
    y_val_oh = keras.utils.to_categorical(y_val, num_classes)

    keras.utils.set_random_seed(SEED)
    head = build_head(num_classes, dropout=args.dropout, feature_dim=x_train.shape[1])
    head.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    head_ckpt = exp_dir / "head_best.keras"
    callbacks = [
        ModelCheckpoint(str(head_ckpt), monitor="val_accuracy", save_best_only=True, verbose=1),
        CSVLogger(str(exp_dir / "history.csv"), append=False),
    ]

    history = head.fit(
        x_train,
        y_train_oh,
        validation_data=(x_val, y_val_oh),
        epochs=args.epochs,
        batch_size=args.head_batch_size,
        callbacks=callbacks,
    )

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history.history, f, indent=2)

    # najbolja glava po val_accuracy, pa sklapanje punog modela slika -> softmax,
    # jer evaluate.py ucitava model koji prima slike, a ne 2048-dim vektore
    best_head = keras.models.load_model(head_ckpt)
    extractor = build_feature_extractor(args.image_size)
    full = keras.Model(
        extractor.input, best_head(extractor.output), name="transfer_dense"
    )
    full.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    full.save(exp_dir / "best.keras")

    meta = {
        "model": "transfer_dense",
        "num_classes": num_classes,
        "image_size": args.image_size,
        "batch_size": args.head_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "dropout": args.dropout,
        "params": int(full.count_params()),
        "trainable_params": int(best_head.count_params()),
        "cached_features": True,
        "augmentation": False,
    }
    with open(exp_dir / "run_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    best_val = max(history.history["val_accuracy"])
    print()
    print("najbolji val_accuracy:", round(float(best_val), 4))
    print("pun model:", exp_dir / "best.keras")
    print("run_config:", exp_dir / "run_config.json")


def main():
    args = parse_args()
    default_name = "transfer_finetune" if args.stage == "finetune" else "transfer_dense"
    exp_dir = Path(args.exp_dir).resolve() if args.exp_dir else ROOT / "experiments" / default_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "finetune":
        # kes i naucena glava zive u dense folderu, rezultati fine-tune-a u svom
        dense_dir = Path(args.dense_dir).resolve() if args.dense_dir else ROOT / "experiments" / "transfer_dense"
        finetune(exp_dir, args, dense_dir)
        return

    names = class_names()
    if args.stage in ("cache", "all"):
        names = cache_features(
            exp_dir, args.image_size, args.batch_size, args.limit, args.recompute
        )
    if args.stage in ("dense", "all"):
        train_head(exp_dir, args, names)


if __name__ == "__main__":
    main()
