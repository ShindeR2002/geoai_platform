import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.datasets.dataset_splitter import split_dataset_unified

def main():
    output_dir = root_dir / "outputs" / "scientific_investigation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = ["ps10_sentinel_v1", "dholera_sentinel_v1"]
    records = []
    
    # Enable matplotlib headless mode
    plt.switch_backend('Agg')
    
    for dataset_id in datasets:
        print(f"Loading dataset: {dataset_id}...")
        dataset = DatasetRegistry.load_dataset(dataset_id, configs_dir=str(root_dir / "configs"))
        X, y = dataset.X, dataset.y
        
        # Split using split_dataset_unified
        split_res = split_dataset_unified(
            X, y, dataset_id,
            split_policy="spatial",
            patch_size=15
        )
        
        splits = {
            "train": split_res.y_train,
            "val": split_res.y_val,
            "test": split_res.y_test
        }
        
        for split_name, split_y in splits.items():
            if split_y is None:
                continue
            total = len(split_y)
            changed = int((split_y == 1).sum())
            unchanged = int((split_y == 0).sum())
            pct_changed = changed / total if total > 0 else 0.0
            pct_unchanged = unchanged / total if total > 0 else 0.0
            
            records.append({
                "dataset_id": dataset_id,
                "split": split_name,
                "total_pixels": total,
                "changed_pixels": changed,
                "unchanged_pixels": unchanged,
                "pct_changed": pct_changed,
                "pct_unchanged": pct_unchanged
            })
            
    df = pd.DataFrame(records)
    
    # Save CSV
    csv_path = output_dir / "dataset_statistics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset statistics CSV to {csv_path}")
    
    # Generate Markdown table
    md_path = output_dir / "dataset_statistics.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Split Statistics\n\n")
        f.write("This report presents the pixel counts and class distribution statistics for the `PS10` and `Dholera` AOIs under the unified spatial block-splitting scheme.\n\n")
        
        # Custom helper to format DF as markdown table
        f.write("| Dataset ID | Split | Total Pixels | Changed Pixels | Unchanged Pixels | % Changed | % Unchanged |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for _, row in df.iterrows():
            f.write(f"| `{row['dataset_id']}` | **{row['split']}** | {row['total_pixels']:,} | {row['changed_pixels']:,} | {row['unchanged_pixels']:,} | {row['pct_changed']*100:.2f}% | {row['pct_unchanged']*100:.2f}% |\n")
        
        f.write("\n> [!NOTE]\n")
        f.write("> Since a spatial block-splitting scheme is employed, the split pixel coordinates are deterministic and invariant across random seeds. Thus, per-seed class distributions remain identical.\n")
    print(f"Saved dataset statistics MD report to {md_path}")
    
    # Generate LaTeX table
    tex_path = output_dir / "publication_dataset_statistics.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% Auto-generated publication table for dataset statistics\n")
        f.write("\\begin{table}[ht]\n\\centering\n\\caption{Pixel Class Distributions across Splits for Study AOIs}\n")
        f.write("\\begin{tabular}{llrrrrr}\n\\hline\n")
        f.write("Dataset ID & Split & Total Pixels & Changed & Unchanged & \\% Changed & \\% Unchanged \\\\\n\\hline\n")
        for _, row in df.iterrows():
            f.write(f"{row['dataset_id'].replace('_', '\\_')} & {row['split']} & {row['total_pixels']:,} & {row['changed_pixels']:,} & {row['unchanged_pixels']:,} & {row['pct_changed']*100:.2f}\\% & {row['pct_unchanged']*100:.2f}\\% \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\label{tab:dataset_stats}\n\\end{table}\n")
    print(f"Saved dataset statistics LaTeX table to {tex_path}")
    
    # Generate PNG bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    for idx, dataset_id in enumerate(datasets):
        ax = axes[idx]
        df_ds = df[df["dataset_id"] == dataset_id]
        
        labels = df_ds["split"].tolist()
        unchanged_vals = df_ds["unchanged_pixels"].tolist()
        changed_vals = df_ds["changed_pixels"].tolist()
        
        width = 0.55
        
        ax.bar(labels, unchanged_vals, width, label="Unchanged (No Change)", color="#34495e")
        ax.bar(labels, changed_vals, width, bottom=unchanged_vals, label="Changed (Change)", color="#e74c3c")
        
        ax.set_title(f"Class Distribution: {dataset_id.split('_')[0].upper()}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Pixel Count", fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.legend(loc="upper right")
        
        # Add labels on top of the bars showing % changed
        for i, row in enumerate(df_ds.itertuples()):
            y_pos = row.total_pixels
            ax.text(i, y_pos + (y_pos * 0.01), f"{row.pct_changed*100:.2f}% Change", ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
            
    plt.tight_layout()
    plot_path = output_dir / "class_distribution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved class distribution PNG to {plot_path}")
    
    # Generate class distribution report MD
    report_path = output_dir / "class_distribution_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Class Distribution and Imbalance Report\n\n")
        f.write("## 1. Summary of Spatial Class Imbalance\n\n")
        f.write("Anthropogenic change detection is characterized by high class imbalance because the area where change actually occurs is small compared to unchanged regions. The statistics for the study sites are:\n\n")
        
        for dataset_id in datasets:
            df_ds = df[df["dataset_id"] == dataset_id]
            test_row = df_ds[df_ds["split"] == "test"].iloc[0]
            f.write(f"### {dataset_id.split('_')[0].upper()} AOI\n")
            f.write(f"- **Total test split pixels:** {test_row['total_pixels']:,}\n")
            f.write(f"- **Change pixels (Class 1):** {test_row['changed_pixels']:,} ({test_row['pct_changed']*100:.2f}%)\n")
            f.write(f"- **No-Change pixels (Class 0):** {test_row['unchanged_pixels']:,} ({test_row['pct_unchanged']*100:.2f}%)\n\n")
            
        f.write("## 2. Impact on Model Performance\n\n")
        f.write("Class imbalance directly explains why deep learning models using unweighted standard cross-entropy loss optimize for the majority class. Models can achieve $>90\\%$ accuracy by simply predicting the negative class everywhere. To achieve high F1 and IoU, models must optimize decision thresholds to shift sensitivity or employ weighted loss functions.\n")
    print(f"Saved class distribution analysis report to {report_path}")

if __name__ == "__main__":
    main()
