# Treniranje VGG-lite modela

import argparse
import json
from pathlib import Path

from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from dataset import IMAGE_SIZE, make_generators
from models import build_vgglite

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "experiments" / "custom_vgglite"


def parse_args():
    p = argparse.ArgumentParser(description="Treniranje VGG-lite na Pandora18K")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--val-steps", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_iter, val_iter, test_iter, names = make_generators(
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    model = build_vgglite(num_classes=len(names), input_size=args.image_size)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = OUT_DIR / "best.keras"
    history_csv = OUT_DIR / "history.csv"

    callbacks = [
        # cuva tezine sa najboljom val_accuracy
        ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # metrike po epohi u CSV
        CSVLogger(str(history_csv), append=False),
    ]

    fit_kwargs = dict(
        x=train_iter,
        validation_data=val_iter,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    if args.steps is not None:
        fit_kwargs["steps_per_epoch"] = args.steps
    if args.val_steps is not None:
        fit_kwargs["validation_steps"] = args.val_steps

    history = model.fit(**fit_kwargs)

    history_path = OUT_DIR / "history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)

    # stanje posle poslednje epohe
    model.save(OUT_DIR / "last.keras")

    meta = {
        "num_classes": len(names),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "steps": args.steps,
        "val_steps": args.val_steps,
        "params": int(model.count_params()),
    }
    with open(OUT_DIR / "run_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("gotovo.")
    print("best model:", ckpt_path)
    print("history:", history_csv, history_path)


if __name__ == "__main__":
    main()
