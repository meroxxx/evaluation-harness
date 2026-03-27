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
                label=f'Avg: {avg:.4f}', alpha=0.7)
    
    plt.title(f"{metric.upper()} Comparison", fontsize=14, fontweight='bold')
    plt.ylabel(metric)
    

    current_vals = df_sorted[metric].dropna()
    if len(current_vals) > 1:
        min_v, max_v = current_vals.min(), current_vals.max()

        if max_v - min_v < 0.1 and max_v > 0.7:
            padding = (max_v - min_v) 
            plt.ylim(max(0, min_v - padding), min(1.0, max_v + padding * 0.5))
        else:
            plt.ylim(0, 1) 
        
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, 
                f"{height:.4f}", ha="center", va="bottom", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"outputs/plots/{metric}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: outputs/plots/{metric}.png")

available_metrics = [m for m in metrics if m in df.columns]
if len(available_metrics) >= 2:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    table_data = []
    for _, row in df.iterrows():
        row_data = [str(row['model'])]
        for m in available_metrics:
            row_data.append(f"{row[m]:.4f}")
        table_data.append(row_data)
    
    col_labels = ['Model'] + [m.upper() for m in available_metrics]
    
    n_cols = len(col_labels)
    col_colors = [colors[i % len(colors)] for i in range(n_cols)]
    col_widths = [0.15] + [0.14] * len(available_metrics)
    
    table = ax.table(cellText=table_data, colLabels=col_labels,
                    loc='center', cellLoc='center',
                    colColours=col_colors, colWidths=col_widths)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 2.5)
    
    for i in range(len(col_labels)):
        table[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=11)
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            cell.set_edgecolor('white')
            cell.set_linewidth(1.5)
            cell.set_facecolor('#F8F9FA' if i % 2 == 0 else '#FFFFFF')
    
    plt.title('Model Performance Comparison', fontsize=15, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.savefig("outputs/plots/comparison_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/plots/comparison_table.png")

if len(available_metrics) >= 2 and len(df) >= 2:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('ML Model Evaluation Dashboard', fontsize=16, fontweight='bold', y=0.98)
    
    n_plots = min(len(available_metrics), 5)
    
    for idx, metric in enumerate(available_metrics[:n_plots]):
        ax = fig.add_subplot(2, 3, idx + 1)
        df_sorted = df.sort_values(by=metric, ascending=True)
        
        bars = ax.barh(df_sorted["model"], df_sorted[metric],
                      color=colors[idx % len(colors)], edgecolor='white', linewidth=2)
        
        avg = df_sorted[metric].mean()
        ax.axvline(x=avg, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        for i, (idx_row, row) in enumerate(df_sorted.iterrows()):
            ax.text(row[metric] +0.02, i, f'{row[metric]:.4f}',
                   va='center', fontsize=9, fontweight='bold',clip_on=True)
        

        vals = df_sorted[metric].dropna()
        if len(vals) > 1:
            min_v, max_v = vals.min(), vals.max()
            span = max_v - min_v
            
            if span < 0.05:
                pad = span * 5 if span > 0 else 0.02
                ax.set_xlim(
                    max(0, min_v - pad), 
                    min(1.001, max_v + pad * 1.5)  
                )
            else:
                ax.set_xlim(0, 1.1)
        else:
            ax.set_xlim(0, 1.1)

        ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlabel('Score', fontsize=9)
        ax.tick_params(labelsize=9)
    
    if n_plots < 6:
        ax_table = fig.add_subplot(2, 3, n_plots + 1)
        ax_table.axis('off')
        
        table_data = []
        for _, row in df.iterrows():
            row_data = [str(row['model'])[:15]]
            for m in available_metrics:
                row_data.append(f"{row[m]:.4f}")
            table_data.append(row_data)
        
        col_labels = ['Model'] + [m[:4].upper() for m in available_metrics]
        
        n_cols = len(col_labels)
        col_colors = [colors[i % len(colors)] for i in range(n_cols)]
        
        table = ax_table.table(cellText=table_data, colLabels=col_labels,
                              loc='center', cellLoc='center',
                              colColours=col_colors)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 2.0)
        
        for i in range(len(col_labels)):
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        ax_table.set_title('Summary', fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("outputs/plots/dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/plots/dashboard.png")

print("Done")