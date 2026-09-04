# Evaluacija sacuvanog modela na test skupu.
# Pokretanje: python src/evaluate.py --model vgglite

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

from dataset import IMAGE_SIZE, make_generators
from models import build_hybrid, build_vgglite

# repo root je jedan nivo iznad src/
ROOT = Path(__file__).resolve().parents[1]

CUSTOM_BUILDERS = {
    "vgglite": build_vgglite,
    "hybrid": build_hybrid,
}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluacija modela na test skupu")
    p.add_argument("--model", default="vgglite", help="ime eksperimenta, npr. vgglite ili hybrid")
    p.add_argument("--exp-dir", default=None, help="rucno zadat folder eksperimenta")
    p.add_argument("--weights", default="best.keras", help="fajl modela unutar exp foldera")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=None)
    return p.parse_args()


def resolve_exp_dir(args):
    # apsolutna putanja, da radi i kad je --exp-dir zadat relativno
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
        return exp_dir if exp_dir.is_absolute() else (Path.cwd() / exp_dir).resolve()
    return ROOT / "experiments" / f"custom_{args.model}"


def exp_name(args, exp_dir):
    # kad je zadat --exp-dir, ime se izvodi iz foldera, da figura ne bi
    # zavrsila pod podrazumevanim --model imenom i pregazila tudju
    if args.exp_dir:
        return exp_dir.name.removeprefix("custom_")
    return args.model


def _rel_to_root(path):
    # eksperiment moze biti i van repoa, pa relative_to nije uvek moguc
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_image_size(args, exp_dir):
    # ako nije zadat, uzima se ista velicina kao na treningu
    if args.image_size is not None:
        return args.image_size
    cfg_path = exp_dir / "run_config.json"
    if cfg_path.is_file():
        with open(cfg_path) as f:
            return json.load(f).get("image_size", IMAGE_SIZE)
    return IMAGE_SIZE


def load_model_for_eval(model_name, model_path, image_size, num_classes):
    try:
        return keras.models.load_model(model_path)
    except (TypeError, ValueError, OSError):
        builder = CUSTOM_BUILDERS.get(model_name)
        if builder is None:
            raise
        model = builder(num_classes=num_classes, input_size=image_size)
        model.load_weights(str(model_path))
        return model


def plot_confusion(cm, names, out_paths, title):
    # normalizovana po redu -> vidi se recall po klasi
    with np.errstate(invalid="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, label="udeo u klasi")

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predikcija")
    ax.set_ylabel("Tacna klasa")
    ax.set_title(title)

    fig.tight_layout()
    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    exp_dir = resolve_exp_dir(args)
    model_name = exp_name(args, exp_dir)
    model_path = exp_dir / args.weights
    if not model_path.is_file():
        raise SystemExit(f"nema modela na putanji: {model_path}")

    image_size = resolve_image_size(args, exp_dir)
    _, _, test_iter, names = make_generators(
        image_size=image_size,
        batch_size=args.batch_size,
    )

    model = load_model_for_eval(args.model, model_path, image_size, len(names))
    # sklopljeni modeli (npr. transfer) ne moraju stici sa compile konfiguracijom,
    # a evaluate() je trazi; za evaluaciju je optimizator nebitan
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    loss, accuracy = model.evaluate(test_iter, verbose=1)

    # shuffle=False u test generatoru, pa se redosled poklapa sa labelama
    probs = model.predict(test_iter, verbose=1)
    y_pred = probs.argmax(axis=1)
    y_true = test_iter.classes

    report = classification_report(
        y_true, y_pred, target_names=names, output_dict=True, zero_division=0
    )
    report_txt = classification_report(
        y_true, y_pred, target_names=names, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(names)))

    metrics = {
        "model": model_name,
        "weights": _rel_to_root(model_path),
        "image_size": image_size,
        "params": int(model.count_params()),
        "test_accuracy": float(accuracy),
        "test_loss": float(loss),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "num_test_images": int(len(y_true)),
    }

    with open(exp_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(exp_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(exp_dir / "classification_report.txt", "w") as f:
        f.write(report_txt + "\n")
    np.savetxt(exp_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")

    plot_confusion(
        cm,
        names,
        [
            exp_dir / "confusion_matrix.png",
            ROOT / "reports" / "figures" / f"{model_name}_confusion.png",
        ],
        f"Matrica konfuzije — {model_name}",
    )

    print()
    print(report_txt)
    print("test accuracy:", round(metrics["test_accuracy"], 4))
    print("test loss:", round(metrics["test_loss"], 4))
    print("macro F1:", round(metrics["macro_f1"], 4))
    print("sacuvano u:", exp_dir)


if __name__ == "__main__":
    main()
