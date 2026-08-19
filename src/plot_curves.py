# Crta accuracy/loss krive iz history.csv nekog eksperimenta.
# Pokretanje: python src/plot_curves.py --model vgglite

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# repo root je jedan nivo iznad src/
ROOT = Path(__file__).resolve().parents[1]

# eksperimenti custom modela stoje pod experiments/custom_<model>,
# transfer modeli pod experiments/<model>
CUSTOM_MODELS = ("vgglite", "hybrid")

TITLES = {
    "vgglite": "VGG-lite",
    "hybrid": "Hibrid",
    "transfer_dense": "InceptionV3 (dense glava)",
    "transfer_finetune": "InceptionV3 (fine-tune)",
}


def parse_args():
    p = argparse.ArgumentParser(description="Krive ucenja iz history.csv")
    p.add_argument("--model", default="vgglite")
    p.add_argument("--exp-dir", default=None, help="rucno zadat folder eksperimenta")
    return p.parse_args()


def resolve_exp_dir(args):
    if args.exp_dir:
        return Path(args.exp_dir)
    if args.model in CUSTOM_MODELS:
        return ROOT / "experiments" / f"custom_{args.model}"
    return ROOT / "experiments" / args.model


def main():
    args = parse_args()
    exp_dir = resolve_exp_dir(args)
    hist_path = exp_dir / "history.csv"
    if not hist_path.is_file():
        raise SystemExit(f"nema history.csv na putanji: {hist_path}")

    title = TITLES.get(args.model, args.model)
    out_exp = exp_dir / "curves.png"
    out_fig = ROOT / "reports" / "figures" / f"{args.model}_curves.png"

    hist = pd.read_csv(hist_path)
    epochs = range(1, len(hist) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, hist["accuracy"], label="train")
    axes[0].plot(epochs, hist["val_accuracy"], label="val")
    axes[0].set_xlabel("Epoha")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title(f"{title} accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, hist["loss"], label="train")
    axes[1].plot(epochs, hist["val_loss"], label="val")
    axes[1].set_xlabel("Epoha")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{title} loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_exp, dpi=150)
    fig.savefig(out_fig, dpi=150)
    plt.close()
    print("sacuvano:", out_exp)
    print("sacuvano:", out_fig)


if __name__ == "__main__":
    main()
