import os
from pathlib import Path
from typing import Dict, Any

class CardsGenerator:
    """Generates Markdown dataset cards and experiment cards detailing metrics, capabilities, and configurations."""
    
    def __init__(self, dataset_output_dir: str = "outputs/datasets/cards", experiment_output_dir: str = "outputs/experiments/cards"):
        self.dataset_dir = Path(dataset_output_dir)
        self.experiment_dir = Path(experiment_output_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def generate_dataset_card(self, dataset_id: str, meta: Any, stats: Dict[str, Any], quality_score: float) -> str:
        """Compile a comprehensive Dataset Card with stats, citations, and recommended models."""
        caps = meta.capabilities.capabilities
        
        card_content = f"""# 📇 Dataset Card: {meta.aoi_name}

## 📝 Description
- **Dataset ID**: `{dataset_id}`
- **Sensor Modality**: `{meta.sensor_eo}`
- **Resolution**: `{meta.spatial_resolution_m}m` per pixel
- **Target Coordinate Reference System (CRS)**: `{meta.crs}`
- **Licensing Model**: `{meta.licensing}`

## 🔬 bands and Class Specifications
- **Bands**: `[Red, Green, Blue, NIR, SWIR]`
- **Classes**: `[0: No-Change, 1: Change]`
- **Evaluation Status**: `{"Multiclass Supported" if caps.get("Multiclass") else "Binary Target Change Detection"}`

## 🛰️ Geographic footprint
- **Bounding Box Zone**: `{meta.region or "Unknown region"}, {meta.country or "Global"}`
- **Climate Context**: `{meta.climate_zone or "Temperate"}`

## 🏆 Capabilities Alignment Badges
- **Classical ML**: `{"🟢 Supported" if caps.get("Classical_ML") else "🔴 Unsupported"}`
- **CNN Architectures**: `{"🟢 Supported" if caps.get("CNN") else "🔴 Unsupported"}`
- **Vision Transformers**: `{"🟢 Supported" if caps.get("Transformer") else "🔴 Unsupported"}`
- **SAR Target Compatibility**: `{"🟢 Supported" if caps.get("SAR") else "🔴 Unsupported"}`

## 🛡️ Diagnostics Diagnostics
- **Calculated Quality Score**: `{quality_score:.1f} / 100`
- **Spatial Autocorrelation (Moran's I)**: `{stats.get("spatial_autocorrelation_moran_i", 0.0):.4f}`
- **Average Cloud Ratio**: `{stats.get("mean_cloud_cover_ratio", 0.0)*100:.1f}%`

## 📄 BibTeX Citation Reference
```latex
{meta.bibtex}
```
"""
        card_path = self.dataset_dir / f"{dataset_id}_card.md"
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card_content)
            
        return str(card_path)

    def generate_experiment_card(self, run_id: str, run_details: Dict[str, Any], metrics: Dict[str, Any]) -> str:
        """Compile an individual Experiment Card logging execution settings and plots."""
        card_content = f"""# 📇 Experiment Card: {run_id}

## 🧪 run parameters
- **Experiment Slug**: `{run_details.get("experiment_id", "N/A")}`
- **Model ID**: `{run_details.get("model_id", "N/A")}`
- **Dataset ID**: `{run_details.get("dataset_id", "N/A")}`
- **Preprocessing Pipeline**: `{run_details.get("preprocessing", "standard")}`
- **Random Seed Lock**: `{run_details.get("seed", 42)}`
- **Device Mode**: `{run_details.get("device_mode", "GPU CUDA")}`

## 📊 Performance Statistics
- **Jaccard Pixel IoU**: `{metrics.get("iou", 0.0):.4f}`
- **Boundary IoU**: `{metrics.get("boundary_iou", 0.0):.4f}`
- **F1 Score**: `{metrics.get("f1", 0.0):.4f}`
- **Calibration ECE**: `{metrics.get("ece", 0.0):.4f}`
- **Inference Throughput**: `{metrics.get("throughput_pixels_sec", 0.0):.1f} px/sec`
- **Peak Device Memory**: `{metrics.get("peak_memory_mb", 0.0):.1f} MB`

## 💻 System Provenance
- **Git Commit SHA**: `{run_details.get("git_commit", "N/A")}`
- **Execution Elapsed Time**: `{run_details.get("execution_time_sec", 0.0):.2f} seconds`

## 📝 Observations & Interpretation
{run_details.get("interpretation", "No diagnostic notes registered.")}
"""
        card_path = self.experiment_dir / f"{run_id}_card.md"
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card_content)
            
        return str(card_path)
