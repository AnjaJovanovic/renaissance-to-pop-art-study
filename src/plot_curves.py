# Crta accuracy/loss krive iz experiments/custom_vgglite/history.csv
# Pokretanje: python plot_curves.py

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# repo root je jedan nivo iznad src/
ROOT = Path(__file__).resolve().parents[1]
HIST_PATH = ROOT / "experiments" / "custom_vgglite" / "history.csv"
OUT_EXP = ROOT / "experiments" / "custom_vgglite" / "curves.png"
OUT_FIG = ROOT / "reports" / "figures" / "vgglite_curves.png"


def main():
    hist = pd.read_csv(HIST_PATH)
    epochs = range(1, len(hist) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, hist["accuracy"], label="train")
    axes[0].plot(epochs, hist["val_accuracy"], label="val")
    axes[0].set_xlabel("Epoha")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("VGG-lite accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, hist["loss"], label="train")
    axes[1].plot(epochs, hist["val_loss"], label="val")
    axes[1].set_xlabel("Epoha")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("VGG-lite loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_EXP, dpi=150)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close()
    print("sacuvano:", OUT_EXP)
    print("sacuvano:", OUT_FIG)


if __name__ == "__main__":
    main()
