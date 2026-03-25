import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/plots", exist_ok=True)
df = pd.read_csv("outputs/metrics_comparison.csv")

print(f"Loaded {len(df)} models")

metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
colors = ['#2E86AB', '#F6A143', '#06A77D', '#E83F6F', '#8B5CF6']

for metric in metrics:
    if metric not in df.columns:
        continue
    
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by=metric, ascending=True)
    
    bars = plt.bar(df_sorted["model"], df_sorted[metric], 
                   color=colors[:len(df_sorted)], edgecolor='white', linewidth=2)
    
    avg = df_sorted[metric].mean()
    plt.axhline(y=avg, color='red', linestyle='--', 
                label=f'Avg: {avg:.3f}', alpha=0.7)
    
    plt.title(f"{metric.upper()} Comparison", fontsize=14, fontweight='bold')
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, 
                f"{height:.3f}", ha="center", va="bottom", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{metric}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: outputs/plots/{metric}.png")

print("Done!")