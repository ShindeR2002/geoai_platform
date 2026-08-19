# ChatGPT Review Package — TinyCD Baseline Execution Recovery

> [!NOTE]
> **Reporting Standard Version:** v1.0
> **Repository Commit:** `46f91b1353`
> **Evaluation Timestamp:** 2026-07-10T14:49:28Z
> **Python Version:** 3.12.10
> **Operating System:** Windows

---

## 1. Recovery Execution Evidence

* **Was training resumed or restarted?**: **Resumed** (Option A). The model successfully picked up from the completed state of Epoch 1 and resumed at Epoch 2 (training loop skipped 0 epochs because `epochs: 1` was configured), moving directly to inference.
* **Execution Command**: `.venv\Scripts\python scripts/run_tinycd_quick_profile.py`
* **Execution Start Time**: `2026-07-10 14:37:11`
* **Execution End Time**: `2026-07-10 14:49:28`
* **Exit Status**: `0` (Success)
* **Final Epoch Reached**: Epoch 1 (resumed loop finished immediately)
* **Checkpoint Path**: `outputs/experiments/dl_tinycd_quick_profile/checkpoints/best_model.pt`
* **Prediction Output Locations**:
  * GeoTIFF Change Mask: `outputs/experiments/dl_tinycd_quick_profile/prediction.tif`
  * Vector Shapefile: `outputs/experiments/dl_tinycd_quick_profile/objects.shp`
  * GeoJSON Objects: `outputs/experiments/dl_tinycd_quick_profile/objects.geojson`

---

## 2. Console Log Excerpts

```text
2026-07-10 14:37:11 | INFO     | geoai.experiments.experiment_runner | Starting Experiment: dl_tinycd_quick_profile (Model: tinycd)
2026-07-10 14:37:11 | INFO     | geoai.experiments.experiment_runner | Loading dataset: ps10_sentinel_v1
...
2026-07-10 14:37:20 | INFO     | geoai.models.baselines.dl_wrapper | Resuming training from epoch 2 with best IoU 0.0385
2026-07-10 14:37:20 | INFO     | geoai.models.baselines.dl_wrapper | Beginning training loop on device cpu for 1 epochs.
2026-07-10 14:37:20 | INFO     | geoai.models.baselines.dl_wrapper | Model fitting complete.
2026-07-10 14:37:20 | INFO     | geoai.models.baselines.dl_wrapper | Loading PyTorch model weights from 'outputs\experiments\dl_tinycd_quick_profile\checkpoints\best_model.pt'.
2026-07-10 14:37:20 | INFO     | geoai.experiments.experiment_runner | Model training complete in 5.1973s.
2026-07-10 14:40:18 | INFO     | geoai.experiments.experiment_runner | Generating spatial prediction deliverables...
...
2026-07-10 14:40:19 | INFO     | geoai.models.inference | Running inference — model='tinycd' input_shape=(590, 1450, 18).
2026-07-10 14:40:19 | INFO     | geoai.models.inference | Subsampled inference dataset to 14 columns to match model capabilities.
2026-07-10 14:48:04 | INFO     | geoai.features.dataset | Prediction raster reconstructed — shape=(590, 1450) change=235141 no_change=428442 nan=191917.
2026-07-10 14:48:04 | INFO     | geoai.models.inference | Computing prediction probabilities.
2026-07-10 14:49:28 | INFO     | geoai.models.inference | Inference complete — prediction_map shape=(590, 1450) change_px=235141.
2026-07-10 14:49:28 | INFO     | geoai.experiments.experiment_runner | Experiment completed successfully!
```

---

## 3. Evaluation Metrics (Direct from CSV)

```csv
metric,value
Accuracy,0.6299
Precision,0.0511
Recall,0.3991
F1,0.0906
IoU,0.0474
Dice,0.0906
ROC AUC,0.5045
Average Precision,0.0500
MCC,0.0176
Balanced Accuracy,0.5201
ECE,0.0513
Brier Score,0.1191
```

---

## 4. Runtime & Resource Measurements

* **Fitting Duration (sec)**: `5.1973`
* **Inference Latency (sec)**: `173.7842`
* **Peak RAM Usage (MB)**: `584.50`
* **Model Size (MB)**: `0.0531`
* **Checkpoint File Size (bytes)**: `55,669`

---

## 5. Errors and Resolutions
* **Error**: `ModuleNotFoundError: No module named 'lightgbm'` occurred when executing using the global `python` environment.
* **Resolution**: Re-ran the entry point script using the virtual environment's interpreter (`.venv\Scripts\python`), which contains all standard package dependencies pre-installed. The execution proceeded and finalized successfully.
