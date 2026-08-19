# Platform Architecture

**Version:** 2.0  
**Design principle:** Platform-first. PS10 is an application deployed on the platform, not the platform itself.

---

## Layer Model

```
┌─────────────────────────────────────────────────────────┐
│  Orchestration Layer                                     │
│  run_pipeline.py  ·  configs/                           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  Delivery Layer                                          │
│  api/  ·  dashboard/  ·  exports/  ·  visualization/    │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  Application Layer                                       │
│  pipeline/stage1.py  ·  stage2.py  ·  stage3.py         │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│  Platform Layer                                          │
│  core/  ·  preprocessing/  ·  features/                  │
│  models/  ·  postprocessing/  ·  analysis/               │
│  classification/                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Dependency Graph

Arrows indicate allowed import directions. Reverse imports are prohibited.

```
utils/ ─────────────────────────────────────────────────────┐
core/ (exceptions, logger, config, io, validation)          │
  │                                                          │
  ▼                                                          │
preprocessing/ (registry, eo, sar, indices, raster)         │
  │                                                          │
  ▼                                                          │
features/ (temporal, pseudo_labels, feature_cube, dataset)  │
  │                                                          │
  ▼                                                          │
models/ (base, registry, training, inference, evaluation)   │
  │                                                          │
  ▼                                                          │
postprocessing/ (cleanup, morphology, masks)                │
  │                                                          │
  ▼                                                          │
analysis/ (objects, statistics, geometry, metrics)          │
  │                                                          │
  ▼                                                          │
classification/ (schema, base, object_features,             │
                 classifier, attributes, confidence,        │
                 spatial_stats, explainability)             │
  │                                                          │
  ▼                                                          │
exports/ (geotiff, shapefile, geojson, csv_export,          │
          reports, packaging)                               │
  │                                                          │
  ▼                                                          │
visualization/ (maps, plots, overlays, dashboard_layers)    │
  │                                                          │
  ▼                                                          │
pipeline/ (stage1, stage2, stage3, generalization)          │
  │                                                          │
  ▼                                                          │
api/ + dashboard/                                           │
  │                                                          │
  ▼                                                          │
run_pipeline.py (CLI entry point)         ◄─────────────────┘
```

---

## Module Responsibilities

### `geoai/core/`
Platform foundation. No scientific logic. No pipeline logic.
- `exceptions.py` — typed exception hierarchy
- `logger.py` — structured logging configuration
- `config.py` — YAML loading, validated dataclasses, AOI path resolution
- `io.py` — rasterio read/write wrappers
- `validation.py` — raster compatibility checks (CRS, dimensions, transform)

### `geoai/utils/`
Shared utilities with no internal dependencies (except each other).
- `constants.py` — canonical feature names, enums, PS10 constants
- `geo.py` — coordinate conversion, PS10 filename formatting
- `raster_utils.py` — NumPy array operations, valid pixel mask, contrast stretch
- `hash_utils.py` — MD5 model hashing for PS10 submission

### `geoai/preprocessing/`
Sensor-aware data loading. One responsibility: load rasters into arrays.
- `registry.py` — sensor capability definitions (Sentinel-2, Sentinel-1, LISS-IV)
- `eo.py` — RGB and index band loading
- `sar.py` — SAR VV loading
- `indices.py` — dual-mode index loading/computation
- `raster.py` — alignment, reprojection, resampling

### `geoai/features/`
Feature engineering. Scientifically frozen.
- `temporal.py` — T2−T1 delta computation
- `pseudo_labels.py` — composite change score and threshold labelling
- `feature_cube.py` — 18-band canonical cube assembly
- `dataset.py` — flattening, valid mask, X/y preparation, reconstruction

### `geoai/models/`
ML model interface and operations.
- `base.py` — BaseModel abstract interface
- `registry.py` — model discovery and loading by ID
- `training.py` — RF Enhanced and Baseline training
- `inference.py` — prediction + spatial reconstruction
- `evaluation.py` — Jaccard, accuracy, classification report

### `geoai/postprocessing/`
Binary mask production from prediction maps.
- `cleanup.py` — `nan_to_num → uint8` (canonical Version 1 operation)
- `morphology.py` — optional closing/opening/dilation (disabled by default)

### `geoai/analysis/`
Spatial object analysis.
- `objects.py` — connected component labelling, area filtering
- `statistics.py` — per-object and summary statistics
- `geometry.py` — pixel-to-geo, vectorisation
- `metrics.py` — Jaccard and spatial evaluation metrics

### `geoai/classification/`
Stage 2 semantic classification.
- `schema.py` — ChangeObject dataclass
- `base.py` — BaseClassifier interface
- `object_features.py` — per-object spectral feature extraction
- `classifier.py` — rule-based (production) and ML-based (deferred) classifiers
- `attributes.py` — ChangeObject population orchestration
- `confidence.py` — evidence-based confidence scoring
- `spatial_stats.py` — class-wise statistics
- `explainability.py` — SHAP feature attribution

### `geoai/exports/`
Output file generation.
- `geotiff.py` — binary and classified change mask GeoTIFF
- `shapefile.py` — Stage 1 and Stage 2 shapefile + GeoJSON
- `csv_export.py` — object statistics and class statistics CSV
- `reports.py` — Stage 1 and Stage 2 text reports
- `packaging.py` — PS10 submission zip assembly

### `geoai/pipeline/`
Workflow orchestration. No scientific logic — delegates to platform modules.
- `stage1.py` — end-to-end Stage 1 change detection
- `stage2.py` — end-to-end Stage 2 object classification
- `stage3.py` — batch multi-AOI Stage 1 + Stage 2
- `generalization.py` — generalisation test runner (NB09 equivalent)

### `geoai/api/`
REST API (FastAPI).
- `router.py` — app factory, middleware, route registration
- `schemas.py` — Pydantic request/response models
- `endpoints/` — one file per endpoint group

### `geoai/dashboard/`
Dashboard backend integration.
- `project_manager.py` — project manifest CRUD, run history
- `layer_manager.py` — raster/vector layer registry
- `report_generator.py` — Markdown project reports

---

## Extensibility Patterns

### Adding a new AOI
1. Add an entry to `configs/processing.yaml` under `aois:`.
2. Place raster files at the configured `data_dir`.
3. Run `python run_pipeline.py --stage 1 --aoi NewAOI`.

### Adding a new model
1. Implement `BaseModel` in `geoai/models/`.
2. Add an entry to `MODEL_REGISTRY` in `geoai/models/registry.py`.
3. Update `configs/model.yaml` with the new `model_id` and `model_path`.

### Adding a new sensor
1. Add a `SensorSpec` entry to `SENSOR_REGISTRY` in `geoai/preprocessing/registry.py`.
2. Update `configs/processing.yaml` AOI entries to reference the new sensor ID.

### Adding a new export format
1. Create a module in `geoai/exports/`.
2. Call it from `geoai/pipeline/stage1.py` or `stage2.py`.

No other source-code changes are required for any of the above.
