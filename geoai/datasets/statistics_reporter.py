import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
from geoai.datasets.dataset_schema import UnifiedChangeDetectionDataset

class DatasetStatisticsReporter:
    """Calculates class distributions, Moran's I spatial autocorrelation, and outputs previews."""
    
    @staticmethod
    def compute_morans_i(label_grid: np.ndarray) -> float:
        """Calculate a grid-based spatial autocorrelation Moran's I index for a binary label map."""
        try:
            flat = label_grid.astype(np.float32).ravel()
            mean = np.mean(flat)
            denom = np.sum((flat - mean) ** 2)
            if denom == 0:
                return 0.0
                
            # Shift label grid horizontally and vertically to simulate neighbor connectivity weights
            shifted_h = np.zeros_like(label_grid, dtype=np.float32)
            shifted_h[:-1, :] = label_grid[1:, :]
            
            shifted_w = np.zeros_like(label_grid, dtype=np.float32)
            shifted_w[:, :-1] = label_grid[:, 1:]
            
            num = np.sum((label_grid - mean) * (shifted_h - mean)) + np.sum((label_grid - mean) * (shifted_w - mean))
            # Normalize by 2 times connectivity to keep bounds around [-1, 1]
            return float(num / (2.0 * denom + 1e-8))
        except Exception:
            return 0.0

    @staticmethod
    def generate_report(dataset: UnifiedChangeDetectionDataset, output_dir: str = "outputs/datasets") -> Dict[str, Any]:
        """Compute pixel stats, class ratios, spatial autocorrelation, and save preview composites."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        meta = dataset.get_metadata()
        dataset_id = meta.dataset_id
        
        total_pixels = 0
        change_pixels = 0
        no_change_pixels = 0
        moran_indices = []
        
        # Take a representative subset for stats if dataset is huge
        limit = min(len(dataset), 20)
        
        for idx in range(limit):
            t1, t2, label = dataset.get_pair(idx)
            total_pixels += label.size
            c_pix = np.sum(label == 1)
            change_pixels += c_pix
            no_change_pixels += np.sum(label == 0)
            
            if c_pix > 0:
                moran_indices.append(DatasetStatisticsReporter.compute_morans_i(label))
                
        change_ratio = float(change_pixels / total_pixels) if total_pixels > 0 else 0.0
        no_change_ratio = 1.0 - change_ratio
        mean_moran = float(np.mean(moran_indices)) if moran_indices else 0.0
        
        report_data = {
            "dataset_id": dataset_id,
            "total_tiles_evaluated": limit,
            "change_pixel_count": int(change_pixels),
            "no_change_pixel_count": int(no_change_pixels),
            "change_ratio": change_ratio,
            "no_change_ratio": no_change_ratio,
            "spatial_autocorrelation_moran_i": mean_moran,
            "spatial_resolution_m": meta.spatial_resolution_m,
            "sensor": meta.sensor_eo
        }
        
        # Save JSON stats
        with open(out_path / f"{dataset_id}_stats.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
            
        # Generate Preview Composites (using first pair)
        if len(dataset) > 0:
            try:
                t1, t2, label = dataset.get_pair(0)
                
                # Reshape to RGB for plotting
                t1_rgb = np.transpose(t1[:3], (1, 2, 0)) if t1.shape[0] >= 3 else t1[0]
                t2_rgb = np.transpose(t2[:3], (1, 2, 0)) if t2.shape[0] >= 3 else t2[0]
                
                # Clip to [0, 1] for safe plotting
                t1_rgb = np.clip(t1_rgb, 0.0, 1.0)
                t2_rgb = np.clip(t2_rgb, 0.0, 1.0)
                
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                cmap1 = None if t1.shape[0] >= 3 else 'gray'
                cmap2 = None if t2.shape[0] >= 3 else 'gray'
                
                axes[0].imshow(t1_rgb, cmap=cmap1)
                axes[0].set_title("Pre-Change (T1)")
                axes[0].axis("off")
                
                axes[1].imshow(t2_rgb, cmap=cmap2)
                axes[1].set_title("Post-Change (T2)")
                axes[1].axis("off")
                
                # Overlay change mask as translucent red on T2
                overlay = np.copy(t2_rgb)
                if label.ndim == 2:
                    if overlay.ndim == 3:
                        overlay[label == 1] = [1.0, 0.0, 0.0]
                    else:
                        overlay = np.stack([overlay, overlay, overlay], axis=-1)
                        overlay[label == 1] = [1.0, 0.0, 0.0]
                        
                axes[2].imshow(overlay)
                axes[2].set_title("Overlay Mask")
                axes[2].axis("off")
                
                fig.suptitle(f"Dataset Preview: {dataset_id}", fontsize=14)
                plt.tight_layout()
                
                preview_dir = out_path / f"{dataset_id}_previews"
                preview_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(preview_dir / "preview_composite.png", dpi=150)
                plt.close()
            except Exception as e:
                # Silently catch visualization plotting errors during test environments
                pass
                
        return report_data
