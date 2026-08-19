import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List

class CampaignPlotsGenerator:
    """Generates publication-quality figures representing benchmark matrix performance results using pure matplotlib."""
    
    def __init__(self, output_dir: str = "outputs/benchmarks/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.titlesize': 14
        })

    def plot_generalization_heatmap(self, transfer_matrix: pd.DataFrame, metric_name: str = "IoU") -> str:
        """Draw a pairwise transfer heatmap matrix (Source vs Target Domain) using pure matplotlib."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Draw heatmap using imshow
        cax = ax.imshow(transfer_matrix.values, cmap="viridis", aspect="auto")
        fig.colorbar(cax, label=metric_name)
        
        # Set ticks
        ax.set_xticks(np.arange(len(transfer_matrix.columns)))
        ax.set_yticks(np.arange(len(transfer_matrix.index)))
        ax.set_xticklabels(transfer_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(transfer_matrix.index)
        
        # Annotate cell values
        for i in range(len(transfer_matrix.index)):
            for j in range(len(transfer_matrix.columns)):
                val = transfer_matrix.values[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val < 0.65 else "black")
                
        plt.title(f"Cross-AOI Generalization Heatmap ({metric_name})")
        plt.ylabel("Source Domain (Trained On)")
        plt.xlabel("Target Domain (Evaluated On)")
        plt.tight_layout()
        
        path = self.output_dir / "generalization_heatmap.png"
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_accuracy_vs_latency(self, df_metrics: pd.DataFrame) -> str:
        """Draw a scatter plot comparing Accuracy vs Inference Latency using pure matplotlib."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Separate Classical vs Deep Learning
        df_metrics["Type"] = df_metrics["model_id"].apply(
            lambda x: "Deep Learning" if any(w in x for w in ["fc", "siam", "cnn"]) else "Classical"
        )
        
        classical = df_metrics[df_metrics["Type"] == "Classical"]
        dl = df_metrics[df_metrics["Type"] == "Deep Learning"]
        
        ax.scatter(
            classical["throughput_pixels_sec"],
            classical["iou"],
            color="#e74c3c",
            marker="o",
            s=120,
            label="Classical ML"
        )
        
        ax.scatter(
            dl["throughput_pixels_sec"],
            dl["iou"],
            color="#3498db",
            marker="s",
            s=120,
            label="Deep Learning"
        )
        
        # Annotate model names
        for idx, row in df_metrics.iterrows():
            ax.text(
                row["throughput_pixels_sec"] * 1.05,
                row["iou"] - 0.005,
                row["model_id"],
                horizontalalignment='left',
                size='small',
                color='#2c3e50',
                weight='semibold'
            )
            
        plt.xscale("log")
        plt.title("Accuracy (IoU) vs. Inference Throughput Trade-off")
        plt.xlabel("Throughput (pixels / second) - Log Scale")
        plt.ylabel("Intersection over Union (IoU)")
        plt.legend(loc="lower left")
        plt.tight_layout()
        
        path = self.output_dir / "accuracy_vs_latency.png"
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_calibration_curves(self, reliability_data: Dict[str, Dict[str, Any]]) -> str:
        """Draws reliability diagrams (fraction of positives vs mean predicted confidence)."""
        plt.figure(figsize=(7, 7))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        
        # Generate colors dynamically
        cmap = plt.get_cmap("tab10")
        for idx, (model_id, data) in enumerate(reliability_data.items()):
            plt.plot(
                data["confidence"],
                data["accuracy"],
                marker="s",
                linewidth=1.5,
                color=cmap(idx),
                label=model_id
            )
            
        plt.title("Probability Calibration Curves")
        plt.xlabel("Mean Predicted Probability (Confidence)")
        plt.ylabel("Fraction of Positives (Actual Accuracy)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        
        path = self.output_dir / "calibration_reliability.png"
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_radar_chart(self, df_radar: pd.DataFrame) -> str:
        """Generates radar chart matching metrics across models using polar axes."""
        categories = list(df_radar.columns[1:])
        N = len(categories)
        
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        cmap = plt.get_cmap("Set2")
        
        for idx, row in df_radar.iterrows():
            values = list(row[1:])
            values += values[:1]
            ax.plot(angles, values, linewidth=1.5, linestyle='solid', color=cmap(idx), label=row['Model'])
            ax.fill(angles, values, color=cmap(idx), alpha=0.1)
            
        plt.xticks(angles[:-1], categories, size=10)
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
        plt.ylim(0, 1.1)
        
        plt.title("Multidimensional Performance Alignment", y=1.08)
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        
        path = self.output_dir / "radar_alignment.png"
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)
