import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("custom_vgglite", "VGG-lite"),
    ("custom_hybrid", "Hibrid"),
    ("transfer_dense", "Transfer dense"),
    ("transfer_finetune", "Transfer fine-tune"),
]


def main():
    names = []
    accs = []
    for folder, label in MODELS:
        path = ROOT / "experiments" / folder / "test_metrics.json"
        with open(path) as f:
            m = json.load(f)
        names.append(label)
        accs.append(100.0 * float(m["test_accuracy"]))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, accs, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 70)
    ax.axhline(5.56, color="gray", linestyle="--", linewidth=1, label="nasumicno (5.56%)")
    ax.set_title("Poredjenje modela - test accuracy")
    ax.legend(loc="upper left")

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out = ROOT / "reports" / "figures" / "model_comparison_test_acc.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("sacuvano:", out)


if __name__ == "__main__":
    main()
