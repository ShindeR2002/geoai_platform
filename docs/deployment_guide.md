# Deployment Guide

## Requirements

| Component | Minimum | PS10 Offline Evaluation Spec |
|-----------|---------|------------------------------|
| OS | Ubuntu 20.04+ | Ubuntu 24.04 LTS |
| CPU cores | 4 | 48+ cores |
| RAM | 16 GB | 256+ GB |
| GPU | None (RF is CPU-only) | 40 GB (for future DL models) |
| Python | 3.10+ | 3.10+ |
| Storage | 50 GB | 100+ GB |

---

## Installation

```bash
# 1. Clone repository
git clone <repo-url> && cd geoai_platform

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Migrate production model
cp /path/to/Version1/outputs/models/rf_enhanced.pkl models/rf/rf_enhanced.pkl

# 4. Verify installation
python -m pytest tests/unit/ -q
# Expected: 89 passed, 0 failed
```

---

## Data Layout

Place raster files at the paths defined in `configs/processing.yaml`:

```
data/
└── raw/
    ├── PS10/
    │   └── EO/
    │       ├── PS10_RGB_2021.tif
    │       ├── PS10_RGB_2024.tif
    │       ├── PS10_S1_VV_2021.tif
    │       ├── PS10_S1_VV_2024.tif
    │       ├── PS10_NDVI_2021.tif
    │       ├── PS10_NDVI_2024.tif
    │       ├── PS10_NDBI_2021.tif
    │       ├── PS10_NDBI_2024.tif
    │       ├── PS10_NDWI_2021.tif
    │       └── PS10_NDWI_2024.tif
    └── Dholera/
        └── EO/
            └── (same pattern with Dholera_ prefix)
```

---

## PS10 Online Submission Workflow

On 31 October 2025 at 12:00 hrs, the PS10 organisers will release the shortlisting dataset scene IDs and reference coordinates.

**Step 1:** Download the shortlisting dataset imagery from the Copernicus and Bhoonidhi portals.

**Step 2:** Update `configs/processing.yaml` — set `reference_lat` and `reference_lon` for the PS10 AOI from the official dataset table.

**Step 3:** Run Stage 1 inference:
```bash
python run_pipeline.py --stage 1 --aoi PS10
```

**Step 4:** Locate the submission package:
```
outputs/{run_id}/exports/PS10_31-Oct-2025_*.zip
```

**Step 5:** Verify the MD5 hash file is present:
```
outputs/{run_id}/exports/rf_enhanced_md5.txt
```

**Step 6:** Submit both the zip and the MD5 hash to the PS10 website before 16:00 hrs.

---

## PS10 Offline Evaluation Preparation

The offline evaluation verifies the model hash and runs inference on the organiser's hardware (Ubuntu 24.04, 48+ cores, 256+ GB RAM, 40 GB GPU).

```bash
# Verify model hash matches the submitted hash
python scripts/generate_md5.py --model models/rf/rf_enhanced.pkl --output outputs/

# Test full pipeline runs within time limits
time python run_pipeline.py --stage 1 --aoi PS10
```

---

## Configuration for Production

Key settings to review before production runs:

**`configs/processing.yaml`:**
```yaml
# Set the actual PS10 reference coordinates when released
aois:
  - name: PS10
    reference_lat: 28.1740   # from PS10 dataset table
    reference_lon: 77.6126   # from PS10 dataset table
```

**`configs/logging.yaml`:**
```yaml
level: INFO         # Use WARNING in production for speed
file: outputs/logs/geoai_platform.log
```

**`configs/export.yaml`:**
```yaml
geotiff_compress: lzw  # LZW for binary masks (recommended)
include_geojson: true  # set false to skip GeoJSON if not needed
```

---

## Docker Deployment (Future)

The platform is Docker-ready. All configuration is injectable via YAML files mounted at a configurable path. No hardcoded paths or hostnames exist in the codebase.

```dockerfile
# Example Dockerfile (not yet in repository)
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT ["python", "run_pipeline.py"]
```

```bash
docker run -v /data:/app/data -v /outputs:/app/outputs \
  geoai-platform --stage 1 --aoi PS10
```

Full Docker support is planned for Version 3.

---

## Regression Test Verification

After deploying on a new machine, run the regression tests to confirm scientific parity:

```bash
# Copy V1 reference files
cp /path/to/Version1/outputs/ml_dataset/X_v2_clean.npy \
   tests/regression/v1_reference/
cp /path/to/Version1/outputs/generalization/Dholera_prediction_map.npy \
   tests/regression/v1_reference/
cp /path/to/Version1/outputs/generalization/Dholera_significant_objects.npy \
   tests/regression/v1_reference/

# Run all tests including regression
python -m pytest tests/ -v
# Expected: 100 passed, 0 failed
```
