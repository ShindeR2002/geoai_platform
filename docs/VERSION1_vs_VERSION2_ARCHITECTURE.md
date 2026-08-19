# Version 1 vs Version 2 Architecture Comparison
## Random Forest Enhanced Model for Anthropogenic Change Detection

**Date:** June 23, 2026  
**Comparison:** Version 1 (05_Random_Forest_Enhanced.ipynb) vs Version 2 (geoai_platform)

---

## Executive Summary

Version 1 is a **notebook-based, linear research workflow** that trains, evaluates, and applies a Random Forest model to a single AOI.  
Version 2 is a **modular, production-grade platform** that abstracts the model training pipeline and focuses on inference and multi-AOI processing through a configurable architecture.

---

## 1. Model Training & Serialization

### Version 1 (Research Notebook)
```
1. Load X_v2_clean.npy and y_v2_clean.npy from disk
2. train_test_split(test_size=0.20, random_state=42, stratify=y)
3. RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
4. rf.fit(X_train, y_train)
5. Evaluate: accuracy, classification_report, confusion_matrix
6. Feature importance analysis
7. joblib.dump(rf, "../outputs/models/rf_enhanced.pkl")
```

**Key Characteristics:**
- **Single model instance** trained end-to-end in notebook
- **Direct sklearn RandomForestClassifier** - no abstraction
- **Training parameters hardcoded** in notebook cells
- **Evaluation metrics manual** - no model registry or validation framework
- **Output:** Raw joblib pickle file with no metadata

### Version 2 (Production Platform)
```
1. geoai.models.registry.load_model_from_config(config)
   ├── Looks up model ID → Model class mapping
   ├── Instantiates RFEnhancedModel()
   ├── Calls instance.load(model_path, metadata_path)
   └── Returns loaded model ready for inference

2. geoai.models.inference.run_inference(model, feature_cube)
   ├── Validates feature count (18 required)
   ├── Prepares inference dataset (handles NaN/invalid pixels)
   ├── Calls model.predict() or model.predict_proba()
   └── Reconstructs prediction raster from valid mask

3. Model metadata validation (optional rf_enhanced_meta.json)
   ├── Validates feature_count
   ├── Validates feature_names
   └── Logs warnings for inconsistencies
```

**Key Characteristics:**
- **Model registry pattern** - decoupled model loading from inference
- **BaseModel abstraction** - RFEnhancedModel implements interface
- **Metadata validation** - model.load() checks consistency
- **Feature validation** - enforces 18-feature requirement
- **Dataset handling** - automatic valid pixel masking
- **Output:** Encapsulated RFEnhancedModel instance with contract enforcement

---

## 2. Feature Engineering Pipeline

### Version 1 (Notebook)
```python
# Manual raster loading
ndvi21 = rasterio.open("../data/PS10_NDVI_2021.tif").read(1)
ndvi24 = rasterio.open("../data/PS10_NDVI_2024.tif").read(1)
# ... (7 more rasterio.open calls)

# Manual delta computation
delta_ndvi = ndvi24 - ndvi21
delta_ndbi = ndbi24 - ndbi21
delta_ndwi = ndwi24 - ndwi21
delta_sar  = sar24 - sar21

# Manual valid mask creation
valid_mask = (
    np.isfinite(delta_ndvi) &
    np.isfinite(delta_ndbi) &
    np.isfinite(delta_ndwi) &
    np.isfinite(delta_sar)
)
```

**Data Flow:**
- Individual rasters loaded in notebook cells
- Feature assembly happens inline
- Valid mask computed ad-hoc
- No reusable abstraction

### Version 2 (Platform)
```
geoai/preprocessing/eo.py
├── load_eo_epoch(rgb_path, ndvi_path, ndbi_path, ndwi_path, epoch_label)
│   ├── load_rgb(path) → (red, green, blue, profile)
│   ├── load_index_band(ndvi_path) → (ndvi_array, profile)
│   ├── load_index_band(ndbi_path) → (ndbi_array, profile)
│   └── load_index_band(ndwi_path) → (ndwi_array, profile)
│
geoai/preprocessing/sar.py
├── load_sar_vv(path, epoch_label) → (sar_array, profile)
│
geoai/features/feature_cube.py
├── build_feature_cube(
│   red_t1, green_t1, blue_t1, sar_t1, ndvi_t1, ndbi_t1, ndwi_t1,
│   red_t2, green_t2, blue_t2, sar_t2, ndvi_t2, ndbi_t2, ndwi_t2
│   ) → (feature_cube[H,W,18], valid_mask)
│
geoai/features/temporal.py
├── compute_all_deltas(feature_cube) → (delta_ndvi, delta_ndbi, delta_ndwi, delta_sar)
├── normalize_delta(delta) → [0, 1]
│
geoai/features/dataset.py
├── InferenceDataset(feature_cube)
│   ├── valid_mask creation
│   ├── flatten valid pixels
│   └── prepare X_valid for model.predict()
```

**Data Flow:**
- **Modular preprocessing** - each raster type has dedicated loader
- **Composable functions** - load_eo_epoch, load_sar_vv, build_feature_cube chain
- **Reusable validation** - valid mask logic centralized in InferenceDataset
- **Contract enforcement** - feature_cube always (H, W, 18)

---

## 3. Model Inference & Prediction

### Version 1 (Notebook)
```python
# Single-shot full-AOI prediction
X_full = np.load("../outputs/ml_dataset/X_v2_clean.npy")
pred_full = rf.predict(X_full)  # Returns flat array of predictions

# Manual raster reconstruction
prediction_map = np.zeros(valid_mask.shape, dtype=np.uint8)
prediction_map[valid_mask] = pred_full

# Visualization
plt.imshow(prediction_map, cmap='Reds')
plt.show()
```

**Workflow:**
- Load pre-flattened dataset
- Direct predict() call
- Manual raster reconstruction
- Ad-hoc visualization

### Version 2 (Platform)
```python
# Stage 1 Pipeline (geoai/pipeline/stage1.py)
feature_cube = build_feature_cube(...)
model = load_model_from_config(config)

prediction_map, valid_mask, probability_map = run_inference(
    model=model,
    feature_cube=feature_cube,
    compute_probabilities=False
)

# Automatic steps inside run_inference():
# 1. Create InferenceDataset(feature_cube)
# 2. Extract valid pixels X_valid
# 3. Call model.predict(X_valid)
# 4. Reconstruct spatial raster from flat predictions + valid_mask
# 5. Return (prediction_map, valid_mask, probability_map)

# Postprocessing
binary_mask = to_binary_mask(prediction_map)
significant_regions, label_array, significant_mask = extract_objects(binary_mask)

# Export (automatic)
export_change_mask_geotiff(binary_mask, profile, path)
export_change_shapefile(significant_mask, profile, path)
export_geojson(objects, significant_mask, profile, path)
export_objects_csv(objects, path)
```

**Workflow:**
- Feature cube → inference dataset → predictions → raster reconstruction → postprocessing → exports (all orchestrated)
- Automatic valid pixel handling
- Multi-format export abstraction
- Reproducible pipeline with config-driven parameters

---

## 4. Object Extraction & Classification

### Version 1 (Notebook)
```python
# Not shown in notebook - this is research-phase work
# (Would be in separate notebook or script)
# Presumably:
# - Connected component labeling
# - Area filtering
# - Object statistics computation
# - No semantic classification (that's Stage 2 in V2)
```

**Status:** Model training only; object-level classification is separate research task.

### Version 2 (Platform - Stage 2)
```python
# Stage 2 Pipeline (geoai/pipeline/stage2.py)
s1_result = run_stage1(config, aoi_name)  # Gets Stage 1 result with label_array

objects = build_attributed_change_objects(
    label_array=s1_result.label_array,
    feature_cube=s1_result.feature_cube,
    resolution_m=aoi.resolution_m,
    profile=s1_result.profile
)

# For each object:
# 1. Extract spatial attributes (area, perimeter, centroid, bbox)
# 2. Extract spectral features (mean RGB, SAR, indices per object)
# 3. Compute temporal deltas per object
# 4. Apply rule-based semantic classifier
# 5. Assign confidence scores

classified_objects = classifier.classify(objects)

# Export Stage 2 deliverables
export_change_shapefile(classified_objects, profile, path)
export_geojson(classified_objects, profile, path)
export_objects_csv(classified_objects, path)
```

**Status:** 
- V1: Model training only (object classification out of scope)
- V2: Full two-stage pipeline with semantic classification and confidence scoring

---

## 5. Configuration & Multi-AOI Processing

### Version 1 (Notebook)
```python
# Hardcoded file paths
X = np.load("../outputs/ml_dataset/X_v2_clean.npy")
y = np.load("../outputs/ml_dataset/y_v2_clean.npy")

ndvi21 = rasterio.open("../data/PS10_NDVI_2021.tif").read(1)
ndvi24 = rasterio.open("../data/PS10_NDVI_2024.tif").read(1)
# ... (manual path management)

# Single AOI only
# Multi-AOI processing requires running notebook multiple times
```

**Scalability:** Limited - manual per-AOI execution.

### Version 2 (Platform)
```yaml
# configs/processing.yaml
aois:
  - name: PS10
    t1_year: 2021
    t2_year: 2024
    data_dir: data/raw/PS10/EO
    resolution_m: 10.0
    file_patterns:
      rgb: "{aoi}_RGB_{year}.tif"
      sar_vv: "{aoi}_S1_VV_{year}.tif"
      ndvi: "{aoi}_NDVI_{year}.tif"
      # ...

  - name: Dholera
    t1_year: 2021
    t2_year: 2024
    data_dir: data/raw/Dholera/EO
    # ...
```

```python
# Command-line
python run_pipeline.py --stage all --aoi PS10 Dholera

# Or Stage 3 (batch)
python run_pipeline.py --stage 3
# Processes all configured AOIs sequentially
```

**Scalability:** Config-driven, multi-AOI batch processing built-in.

---

## 6. Export & Deliverables

### Version 1 (Notebook)
```python
# Manual saves
np.save("../outputs/ml_dataset/rf_prediction_map.npy", prediction_map)
joblib.dump(rf, "../outputs/models/rf_enhanced.pkl")

# Visualization only
plt.imshow(rgb)
plt.imshow(masked_prediction_map, cmap="Reds", alpha=1)
plt.show()

# Optional: PS10 submission zip (manual)
# (Not shown in notebook)
```

**Exports:**
- Prediction map (NumPy)
- Trained model (joblib)
- Visualizations (PNG, manual)

### Version 2 (Platform - Stage 1 & 2)
```
Stage 1 Exports:
├── exports/
│   ├── Change_Mask_PS10.tif          (GeoTIFF)
│   ├── Change_Mask_PS10.shp          (Shapefile)
│   ├── Change_Objects_PS10.geojson   (GeoJSON)
│   ├── Object_Statistics_PS10.csv    (CSV)
│   └── rf_enhanced_md5.txt           (Model hash)
├── reports/
│   └── stage1_report.txt
└── figures/
    └── change_overlay_PS10.png

Stage 2 Exports:
├── exports/
│   ├── Classified_Objects_PS10.shp   (Classified shapefile)
│   ├── Classified_Objects_PS10.geojson
│   ├── Objects_PS10.csv              (Object attributes)
│   └── Class_Statistics_PS10.csv     (Per-class summaries)
├── reports/
│   └── stage2_report.txt

PS10 Submission Zip:
└── PS10_23-Jun-2026_GeoAI_Platform.zip
    ├── Change_Mask_*.tif
    ├── Change_Mask_*.shp
    └── rf_enhanced_md5.txt
```

**Exports:**
- Multi-format (GeoTIFF, Shapefile, GeoJSON, CSV)
- Metadata (MD5, reports, statistics)
- Automatic PS10 submission packaging
- Per-AOI run organization (timestamps in folder names)

---

## 7. Testing & Validation

### Version 1 (Notebook)
```python
# Inline evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

# Visualization-based validation
# No automated test suite
```

**Testing:** Manual, visual, notebook-based.

### Version 2 (Platform)
```
tests/
├── unit/
│   ├── test_feature_cube.py          (18 feature ordering, shape)
│   ├── test_objects.py               (connected components, area filtering)
│   ├── test_temporal.py              (delta computation, thresholds)
│   └── test_valid_mask.py            (NaN handling, reconstruction)
├── integration/
│   └── (integration tests)
└── regression/
    └── test_v1_parity.py             (Version 1 compatibility tests)
```

**Results:** 89 passed, 11 skipped  
**Coverage:** Feature engineering, object extraction, temporal analysis, data validation.

---

## 8. Key Architectural Differences

| Aspect | Version 1 | Version 2 |
|--------|-----------|----------|
| **Scope** | Model training (research) | Full production pipeline (inference + multi-stage classification) |
| **Modularity** | Linear notebook workflow | Modular, layered architecture |
| **Configuration** | Hardcoded paths & parameters | YAML config-driven |
| **Multi-AOI** | Manual re-execution | Batch processing via Stage 3 |
| **Model Loading** | Direct joblib.load() | Model registry + abstraction layer |
| **Feature Engineering** | Inline in notebook | Modular preprocessing modules |
| **Inference** | Single predict() call | Orchestrated pipeline with validation |
| **Object Classification** | Not included | Full Stage 2 with rule-based classifier |
| **Exports** | Basic (NPY, PKL) | Multi-format (GeoTIFF, Shapefile, GeoJSON, CSV) |
| **Testing** | Manual visual | Automated test suite (89 tests) |
| **Reproducibility** | Notebook cells | Config files + deterministic pipeline |

---

## 9. How Version 2 Extends Version 1

### What V2 Preserved from V1
✓ Random Forest model with 18 features  
✓ Feature set: RGB, SAR, NDVI, NDBI, NDWI (both years) + deltas  
✓ Model training logic: 100 estimators, random_state=42, sklearn RandomForestClassifier  
✓ Valid pixel masking based on finite deltas  
✓ Prediction map reconstruction from flat predictions  

### What V2 Adds
✓ **Stage 2 Classification:** Rule-based semantic classifier for object-level change types (Building, Land Clearing, Kacha Track, Unknown)  
✓ **Confidence Scoring:** Per-object confidence values based on spectral consistency  
✓ **Multi-AOI Batch Processing:** Config-driven processing of multiple AOIs  
✓ **Model Registry:** Decoupled model loading and management  
✓ **Feature Validation:** Enforces 18-feature contract  
✓ **Multi-Format Export:** GeoTIFF, Shapefile, GeoJSON, CSV with automatic PS10 submission packaging  
✓ **Test Suite:** 89 automated tests covering all major components  
✓ **API Server:** FastAPI REST endpoint for inference  
✓ **Logging & Reporting:** Structured logging, markdown reports per stage  

---

## 10. Model Migration Path (V1 → V2)

### Step 1: Verify Model Compatibility
```python
import joblib
rf = joblib.load("../outputs/models/rf_enhanced.pkl")
print(f"n_estimators: {rf.n_estimators}")  # Should be 100
print(f"n_features_in_: {rf.n_features_in_}")  # Should be 18
print(f"classes_: {rf.classes_}")  # Should be [0, 1]
```

### Step 2: Migrate Model File
```bash
python scripts/migrate_model.py \
  --source /path/to/Version1/outputs/models/rf_enhanced.pkl \
  --verify
```

### Step 3: Verify in V2 Platform
```bash
python run_pipeline.py --stage 1 --aoi Dholera
# Platform will load rf_enhanced.pkl and run end-to-end inference
```

### Step 4: Validate Output
```bash
# Check outputs/ folder for:
# - Change_Mask_*.tif
# - Change_Mask_*.shp
# - Object_Statistics_*.csv
# - Stage reports
```

---

## 11. Summary Table: Workflow Comparison

### Version 1 Workflow
```
Load Data (Notebook) 
  ↓
Train RF Model (Notebook Cell)
  ↓
Evaluate Model (Notebook Cell)
  ↓
Generate Predictions (Notebook Cell)
  ↓
Reconstruct Raster (Notebook Cell)
  ↓
Manual Save (joblib.dump)
  ↓
Visualization (plt.show)
```
**Time to inference:** 1 notebook execution per AOI  
**Outputs:** NPY, PKL, PNG

### Version 2 Workflow
```
Configuration (YAML)
  ↓
Load Rasters (Preprocessing)
  ↓
Build Feature Cube (Feature Engineering)
  ↓
Load Pre-trained Model (Model Registry)
  ↓
Run Inference (Validation + Prediction)
  ↓
Extract Objects (Postprocessing)
  ↓
Stage 2 Classification (Rule-based)
  ↓
Export Multi-Format (Automatic)
  ↓
Generate Reports (Logging)
```
**Time to inference:** 1 command-line call for multiple AOIs  
**Outputs:** GeoTIFF, Shapefile, GeoJSON, CSV, Reports, Logs, Submission ZIP

---

## Conclusion

**Version 1** is a **proof-of-concept notebook** demonstrating RF model training for change detection.  
**Version 2** is a **production platform** that operationalizes the trained model with:
- Multi-AOI batch processing
- Two-stage classification pipeline
- Automated quality control & exports
- API deployment capability
- Comprehensive testing & logging

The current placeholder model in V2 can be replaced with the V1-trained model using the migration script, after which V2 will automatically handle all downstream processing.

---

