# Feature Specification

**Version:** 2.0  
**Status:** Frozen — modification requires model retraining  
**Source:** Version 1 Notebook 05 (NB05), Notebook 09 (NB09)

---

## Canonical 18-Feature List

The feature order below is the single source of truth for the `rf_enhanced.pkl` model.
Column index in the feature matrix corresponds directly to the position in this list.

| Index | Name | Group | Epoch | Description |
|-------|------|-------|-------|-------------|
| 0 | Red_2021 | A | T1 | Red reflectance, 2021 (Band 1 of RGB raster) |
| 1 | Green_2021 | A | T1 | Green reflectance, 2021 (Band 2 of RGB raster) |
| 2 | Blue_2021 | A | T1 | Blue reflectance, 2021 (Band 3 of RGB raster) |
| 3 | SAR_2021 | A | T1 | SAR VV backscatter, 2021 (Band 1 of S1 raster) |
| 4 | NDVI_2021 | A | T1 | NDVI value, 2021 (pre-computed, single-band raster) |
| 5 | NDBI_2021 | A | T1 | NDBI value, 2021 (pre-computed, single-band raster) |
| 6 | NDWI_2021 | A | T1 | NDWI value, 2021 (pre-computed, single-band raster) |
| 7 | Red_2024 | B | T2 | Red reflectance, 2024 |
| 8 | Green_2024 | B | T2 | Green reflectance, 2024 |
| 9 | Blue_2024 | B | T2 | Blue reflectance, 2024 |
| 10 | SAR_2024 | B | T2 | SAR VV backscatter, 2024 |
| 11 | NDVI_2024 | B | T2 | NDVI value, 2024 |
| 12 | NDBI_2024 | B | T2 | NDBI value, 2024 |
| 13 | NDWI_2024 | B | T2 | NDWI value, 2024 |
| 14 | Delta_NDVI | C | Δ | NDVI_2024 − NDVI_2021 (signed) |
| 15 | Delta_NDBI | C | Δ | NDBI_2024 − NDBI_2021 (signed) |
| 16 | Delta_NDWI | C | Δ | NDWI_2024 − NDWI_2021 (signed) |
| 17 | Delta_SAR | C | Δ | SAR_2024 − SAR_2021 (signed) |

**Groups:**
- **A** — Epoch T1 (earlier epoch, 7 features, indices 0–6)
- **B** — Epoch T2 (later epoch, 7 features, indices 7–13)
- **C** — Temporal deltas (4 features, indices 14–17)

---

## Data Types and Value Ranges

| Feature Group | Dtype | Typical Range | NaN Handling |
|--------------|-------|--------------|--------------|
| RGB bands | float32 | [0, 1] (normalised) or raw DN | nan_to_num → 0.0 |
| SAR VV | float32 | varies (dB scale) | nan_to_num → 0.0 |
| Spectral indices | float32 | [−1, 1] | nan_to_num → 0.0 |
| Temporal deltas | float32 | approximately [−2, 2] | derived from NaN-filled inputs |

---

## Valid Pixel Mask

A pixel is **excluded** from the feature matrix if:
1. Any of its 18 feature values is NaN, **OR**
2. All 18 feature values are exactly 0.0

```python
valid = (~np.isnan(X).any(axis=1)) & (~np.all(X == 0, axis=1))
```

This mask is applied identically in both training and inference.

---

## File Naming Convention

| Data Item | Pattern | Example |
|-----------|---------|---------|
| RGB raster | `{AOI}_RGB_{YYYY}.tif` | `Dholera_RGB_2021.tif` |
| SAR VV raster | `{AOI}_S1_VV_{YYYY}.tif` | `Dholera_S1_VV_2024.tif` |
| NDVI raster | `{AOI}_NDVI_{YYYY}.tif` | `PS10_NDVI_2021.tif` |
| NDBI raster | `{AOI}_NDBI_{YYYY}.tif` | `PS10_NDBI_2024.tif` |
| NDWI raster | `{AOI}_NDWI_{YYYY}.tif` | `Dholera_NDWI_2021.tif` |

Patterns are configurable per-AOI in `configs/processing.yaml`.
