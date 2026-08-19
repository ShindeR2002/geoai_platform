# Scientific Methodology

**Source:** Version 1 research notebooks (validated ground truth)  
**Platform module:** `geoai/features/`, `geoai/preprocessing/`, `geoai/models/`

---

## 1. Data Sources

| Sensor | Type | Resolution | Source | AOIs |
|--------|------|-----------|--------|------|
| Sentinel-2 | EO (RGB, indices) | 10 m | Copernicus Open Access Hub | PS10, Dholera |
| Sentinel-1 | SAR (VV) | 10 m | Copernicus Open Access Hub | PS10, Dholera |

Epochs: T1 = 2021, T2 = 2024.

Spectral indices (NDVI, NDBI, NDWI) are **pre-computed** and provided as single-band GeoTIFF files. The platform loads these directly (Mode 1). Mode 2 computation from raw Sentinel-2 bands is available as an optional capability for new datasets.

---

## 2. Spectral Index Definitions

### NDVI — Normalised Difference Vegetation Index
```
NDVI = (NIR - Red) / (NIR + Red)
```
Sentinel-2: NIR = Band 8 (842 nm), Red = Band 4 (665 nm).  
Range: [−1, 1]. High positive values = dense vegetation.

### NDBI — Normalised Difference Built-up Index
```
NDBI = (SWIR - NIR) / (SWIR + NIR)
```
Sentinel-2: SWIR = Band 11 (1610 nm), NIR = Band 8.  
Range: [−1, 1]. Positive values = built-up or impervious surfaces.

### NDWI — Normalised Difference Water Index
```
NDWI = (Green - NIR) / (Green + NIR)
```
Sentinel-2: Green = Band 3 (560 nm), NIR = Band 8.  
Range: [−1, 1]. Positive values = open water.

---

## 3. Feature Engineering

### 3.1 Per-Epoch Feature Assembly (7 features per epoch)

For each epoch, seven band arrays are loaded and NaN-filled:

```
[Red, Green, Blue, SAR_VV, NDVI, NDBI, NDWI]
```

NaN handling (Version 1 canonical):
```python
band = np.nan_to_num(rasterio.read(band_index), nan=0, posinf=0, neginf=0)
```

### 3.2 Temporal Delta Features (4 features)

Simple arithmetic difference (T2 − T1), applied **without normalisation**:

```python
delta_ndvi = ndvi_t2 - ndvi_t1    # Feature index 14
delta_ndbi = ndbi_t2 - ndbi_t1    # Feature index 15
delta_ndwi = ndwi_t2 - ndwi_t1    # Feature index 16
delta_sar  = sar_t2  - sar_t1     # Feature index 17
```

Deltas are **signed** — negative values indicate decrease; positive values indicate increase.

### 3.3 Canonical 18-Feature Cube

Assembled using `np.stack([...], axis=-1)` in strict index order:

| Index | Feature | Source |
|-------|---------|--------|
| 0 | Red_2021 | RGB raster Band 1, T1 |
| 1 | Green_2021 | RGB raster Band 2, T1 |
| 2 | Blue_2021 | RGB raster Band 3, T1 |
| 3 | SAR_2021 | S1 VV raster Band 1, T1 |
| 4 | NDVI_2021 | NDVI raster Band 1, T1 |
| 5 | NDBI_2021 | NDBI raster Band 1, T1 |
| 6 | NDWI_2021 | NDWI raster Band 1, T1 |
| 7 | Red_2024 | RGB raster Band 1, T2 |
| 8 | Green_2024 | RGB raster Band 2, T2 |
| 9 | Blue_2024 | RGB raster Band 3, T2 |
| 10 | SAR_2024 | S1 VV raster Band 1, T2 |
| 11 | NDVI_2024 | NDVI raster Band 1, T2 |
| 12 | NDBI_2024 | NDBI raster Band 1, T2 |
| 13 | NDWI_2024 | NDWI raster Band 1, T2 |
| 14 | Delta_NDVI | NDVI_2024 − NDVI_2021 |
| 15 | Delta_NDBI | NDBI_2024 − NDBI_2021 |
| 16 | Delta_NDWI | NDWI_2024 − NDWI_2021 |
| 17 | Delta_SAR | SAR_2024 − SAR_2021 |

**This order is frozen by `rf_enhanced.pkl`. It must not change.**

### 3.4 Valid Pixel Mask

Canonical compound mask (NB09 Cell 12):

```python
X_flat = feature_cube.reshape(-1, 18)
valid_mask = (
    ~np.isnan(X_flat).any(axis=1)   # no NaN in any feature
) & (
    ~np.all(X_flat == 0, axis=1)    # not an all-zero pixel
)
X_valid = X_flat[valid_mask]
```

Applied identically in **both training and inference**.

---

## 4. Pseudo-Label Generation (Training Mode)

Training labels are generated from the composite change score when no ground truth exists.

### 4.1 Per-Feature Normalisation

```python
delta_norm = |delta| / max(|delta|)   # for each of the 4 delta features
```

### 4.2 Composite Change Score

```python
change_score = (0.35 * delta_sar_norm
              + 0.35 * delta_ndvi_norm
              + 0.20 * delta_ndbi_norm
              + 0.10 * delta_ndwi_norm)
```

Source: Version 1 NB01 Cell 45. Weights are configurable in `configs/processing.yaml`.

### 4.3 Threshold

```python
threshold = mean(change_score) + 1.5 * std(change_score)
change_mask = (change_score > threshold).astype(np.uint8)
```

Source: Version 1 NB01 Cell 51. Sigma multiplier (1.5) is configurable.

---

## 5. Model

### 5.1 RF Enhanced (Production)

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    # class_weight is NOT set — intentional (differs from Baseline)
)
train_test_split(test_size=0.20, random_state=42, stratify=y)
```

Source: Version 1 NB05. Model file: `models/rf/rf_enhanced.pkl` (113 MB).

### 5.2 RF Baseline (Comparison Only)

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",   # only difference from Enhanced
)
```

Uses 14 features (no delta block). Source: Version 1 NB04. Not used in production.

### 5.3 Inference Pattern (NB09 Canonical)

```python
predictions = rf.predict(X_valid)
prediction_flat = np.full(H * W, np.nan, dtype=np.float32)
prediction_flat[valid_mask] = predictions
prediction_map = prediction_flat.reshape(H, W)
```

Invalid pixel positions are NaN in the prediction map and become 0 in the binary mask.

---

## 6. Post-Processing

```python
binary_change = np.nan_to_num(prediction_map, nan=0).astype(np.uint8)
```

Source: Version 1 NB09 Cell 16.

---

## 7. Object Extraction

```python
from skimage.measure import label, regionprops

labels = label(binary_change, connectivity=2)       # 8-connectivity
regions = regionprops(labels)
significant = [r for r in regions if r.area >= 50]  # MIN_OBJECT_SIZE = 50 px
```

Source: Version 1 NB06 Cell 8, NB09 Cell 17-18.

**Parameters (frozen):**
- `connectivity=2` — 8-connectivity (diagonal neighbours included)
- `MIN_OBJECT_SIZE=50` — minimum object area in pixels (configurable default)

---

## 8. Evaluation Metric

Primary metric for the PS10 challenge: **Jaccard Index (Intersection over Union)**

```
Jaccard = TP / (TP + FP + FN)
```

where:
- TP = pixels correctly predicted as change
- FP = pixels predicted as change but actually no-change
- FN = pixels missed (change but predicted as no-change)

Source: PS10 Problem Statement Section 8.a.2.

---

## 9. Version 1 Notebook to Module Mapping

| Notebook | Scientific Content | Platform Module |
|----------|-------------------|-----------------|
| NB01 | Pseudo-label generation, change score | `features/pseudo_labels.py` |
| NB03 | Dataset preparation | `features/dataset.py` |
| NB04 | RF Baseline training | `models/baselines/rf_baseline.py` |
| NB05 | RF Enhanced training | `models/training.py` |
| NB06 | Object analysis | `analysis/objects.py`, `analysis/statistics.py` |
| NB09 | End-to-end inference | `pipeline/stage1.py`, `pipeline/generalization.py` |
| NB10 | Export (empty) | `exports/` — built from scratch |
