import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/metrics_comparison.csv")

print(df)

metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

for metric in metrics:
    plt.figure()
    
    bars = plt.bar(df["model"], df[metric])
    
    plt.title(metric.upper())
    plt.ylabel(metric)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    plt.ylim(0, 1)
    plt.grid(axis="y")
    
    plt.savefig(f"{metric}.png")
    plt.close()

print("Графики сохранены!")