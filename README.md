# GeoAI Platform — Version 2

Production-grade platform for PS10 Anthropogenic Change Detection using Sentinel-2 EO, Sentinel-1 SAR, and Random Forest machine learning.

## Overview

The GeoAI Platform detects man-made changes between two satellite imagery epochs (2021 → 2024) and classifies detected objects into PS10 Stage 2 semantic categories. It is built as a reusable, modular software platform — not a notebook conversion — using Version 1 research notebooks strictly as a source of validated scientific knowledge.

### PS10 Challenge Coverage

| Stage | Capability | Status |
|-------|------------|--------|
| Stage 1 | EO change detection → GeoTIFF + Shapefile | ✅ Complete |
| Stage 2 | Object classification (6 PS10 classes) | ✅ Rule-based complete; ML-ready |
| Stage 3 | Batch multi-AOI + REST API | ✅ Complete |

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd geoai_platform

# Install dependencies
pip install -r requirements.txt

# Optional: install SHAP for Stage 2 explainability
pip install shap
```

### Migrate the production model

Copy `rf_enhanced.pkl` from Version 1 to the platform model directory:

```bash
cp path/to/Version1/outputs/models/rf_enhanced.pkl models/rf/rf_enhanced.pkl
```

Generate and verify the MD5 hash for PS10 submission:

```bash
python scripts/generate_md5.py --model models/rf/rf_enhanced.pkl --output outputs/
```

---

## Configuration

All runtime parameters live in `configs/`. The key files are:

| File | Controls |
|------|----------|
| `configs/processing.yaml` | AOI definitions, thresholds, pseudo-label weights |
| `configs/model.yaml` | Model path, hyperparameters, feature count |
| `configs/features.yaml` | Canonical 18-feature list (frozen) |
| `configs/export.yaml` | Output formats, PS10 filename convention |
| `configs/logging.yaml` | Log level, file output |
| `configs/api.yaml` | API host, port, CORS |

### Adding a new AOI

Add an entry to the `aois` list in `configs/processing.yaml`:

```yaml
aois:
  - name: NewSite
    t1_year: 2021
    t2_year: 2024
    data_dir: data/raw/NewSite/EO
    sensor_eo: sentinel2
    sensor_sar: sentinel1
    resolution_m: 10.0
    index_mode: load          # 'load' = precomputed rasters (default)
    reference_lat: 28.1740    # PS10 dataset reference coordinates
    reference_lon: 77.6126
```

No source code changes are required.

---

## Running the Platform

### Stage 1 — Change Detection

```bash
# Single AOI
python run_pipeline.py --stage 1 --aoi Dholera

# All configured AOIs
python run_pipeline.py --stage 1

# With debug logging
python run_pipeline.py --stage 1 --aoi PS10 --log-level DEBUG
```

### Stage 2 — Object Classification

```bash
python run_pipeline.py --stage 2 --aoi Dholera
```

### All Stages (1 + 2)

```bash
python run_pipeline.py --stage all --aoi Dholera
```

### Stage 3 — Batch Multi-AOI

```bash
python run_pipeline.py --stage 3 --project-id PS10_Batch_001
```

### REST API

```bash
python run_pipeline.py --api
# API available at http://localhost:8000
# Documentation at http://localhost:8000/docs
```

### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard uses the existing `configs/` and `outputs/` folders to drive AOI inspection, pipeline execution, and export downloads.

---

## PS10 Submission

After a Stage 1 run, the PS10-compliant submission package is automatically created at:

```
outputs/{run_id}/exports/PS10_{DD-MMM-YYYY}_{TeamName}.zip
```

The zip contains:
- `Change_Mask_{Lat}_{Long}.tif` — binary GeoTIFF (pixel 1 = change, 0 = no change)
- `Change_Mask_{Lat}_{Long}.shp` — corresponding shapefile
- `rf_enhanced_md5.txt` — MD5 hash of the model file

**Important:** Set `reference_lat` and `reference_lon` in `configs/processing.yaml` for the PS10 AOI once the shortlisting dataset coordinates are released (31 October 2025 at 12:00 hrs).

---

## Output Structure

Each pipeline run produces a timestamped output directory:

```
outputs/{run_id}/
├── exports/
│   ├── Change_Mask_{Lat}_{Long}.tif      ← PS10 raster deliverable
│   ├── Change_Mask_{Lat}_{Long}.shp      ← PS10 vector deliverable
│   ├── Change_Objects_{AOI}.geojson      ← GeoJSON (Stage 1)
│   ├── Object_Statistics_{AOI}.csv       ← Per-object CSV
│   └── PS10_*.zip                        ← Submission package
├── reports/
│   ├── stage1_report.txt
│   └── stage2_report.txt
├── figures/
│   └── change_overlay_{AOI}.png
└── logs/
    └── geoai_platform.log
```

---

## Running Tests

```bash
# All tests
python -m pytest tests/

# Unit tests only
python -m pytest tests/unit/ -v

# Regression tests (requires V1 reference files)
python -m pytest tests/regression/ -v
```

### Populating regression reference files

Copy the Version 1 output arrays to the regression reference directory:

```bash
cp Version1/outputs/ml_dataset/X_v2_clean.npy            tests/regression/v1_reference/
cp Version1/outputs/generalization/Dholera_prediction_map.npy  tests/regression/v1_reference/
cp Version1/outputs/generalization/Dholera_significant_objects.npy tests/regression/v1_reference/
```

Once present, the 11 skipped regression tests will activate and verify numerical parity with Version 1.

---

## Scientific Methodology

The platform implements the methodology validated in Version 1 research notebooks:

**Feature Engineering (18 features, frozen order):**

| Index | Feature | Index | Feature |
|-------|---------|-------|---------|
| 0 | Red_2021 | 7 | Red_2024 |
| 1 | Green_2021 | 8 | Green_2024 |
| 2 | Blue_2021 | 9 | Blue_2024 |
| 3 | SAR_2021 | 10 | SAR_2024 |
| 4 | NDVI_2021 | 11 | NDVI_2024 |
| 5 | NDBI_2021 | 12 | NDBI_2024 |
| 6 | NDWI_2021 | 13 | NDWI_2024 |
| 14 | Delta_NDVI | 15 | Delta_NDBI |
| 16 | Delta_NDWI | 17 | Delta_SAR |

**Model:** `RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)`

**Object Extraction:** `skimage.measure.label(connectivity=2)` → area filter ≥ 50 px

**Evaluation Metric:** Jaccard Index (PS10 primary metric)

See `docs/scientific_methodology.md` for full equations and Version 1 source references.

---

## Architecture

```
Platform Layer       core/, preprocessing/, features/, models/, analysis/
Application Layer    pipeline/stage1.py, stage2.py, stage3.py
Delivery Layer       exports/, visualization/, api/, dashboard/
Orchestration        run_pipeline.py, configs/
```

Dependencies flow strictly downward. No reverse or circular imports are permitted.

See `docs/architecture.md` for the full dependency graph and design rationale.

---

## Version History

| Version | Description |
|---------|-------------|
| 1.0 | Research notebooks (Version 1 archive) |
| 2.0 | Production platform — full PS10 Stage 1/2/3 implementation |
