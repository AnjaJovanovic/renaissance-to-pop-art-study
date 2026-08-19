# Treniranje custom CNN modela (vgglite / hybrid)

import argparse
import json
from pathlib import Path

from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from dataset import IMAGE_SIZE, make_generators
from models import build_hybrid, build_vgglite

# repo root je jedan nivo iznad src/
ROOT = Path(__file__).resolve().parents[1]

BUILDERS = {
    "vgglite": build_vgglite,
    "hybrid": build_hybrid,
}


def parse_args():
    p = argparse.ArgumentParser(description="Treniranje custom CNN na Pandora18K")
    p.add_argument("--model", choices=BUILDERS.keys(), default="vgglite")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--val-steps", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / "experiments" / f"custom_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_iter, val_iter, test_iter, names = make_generators(
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    model = BUILDERS[args.model](num_classes=len(names), input_size=args.image_size)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = out_dir / "best.keras"
    history_csv = out_dir / "history.csv"

    callbacks = [
        # cuva tezine kojr imaju najbolji val_accuracy
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

    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)

    model.save(out_dir / "last.keras")

    meta = {
        "model": args.model,
        "num_classes": len(names),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "steps": args.steps,
        "val_steps": args.val_steps,
        "params": int(model.count_params()),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("gotovo.")
    print("model:", args.model)
    print("best model:", ckpt_path)
    print("history:", history_csv, history_path)


if __name__ == "__main__":
    main()
