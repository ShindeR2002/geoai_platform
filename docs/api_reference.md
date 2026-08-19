# REST API Reference

**Base URL:** `http://{host}:{port}/v1`  
**Default:** `http://localhost:8000/v1`  
**Interactive docs:** `http://localhost:8000/docs`

Start the API server:
```bash
python run_pipeline.py --api
```

---

## Endpoints

### `POST /v1/predict`

Run Stage 1 change detection on a configured AOI.

**Request body:**
```json
{
  "aoi_name": "Dholera",
  "run_id": "optional_custom_id",
  "compute_probabilities": false
}
```

**Response:**
```json
{
  "run_id": "Dholera_20261020_143012",
  "aoi_name": "Dholera",
  "status": "success",
  "change_pixels": 1847,
  "change_area_ha": 18.47,
  "object_count": 23,
  "output_paths": {
    "geotiff": "outputs/Dholera_.../exports/Change_Mask_22.4167_72.1833.tif",
    "shapefile": "outputs/Dholera_.../exports/Change_Mask_22.4167_72.1833.shp"
  }
}
```

---

### `POST /v1/batch_predict`

Run Stage 1 + Stage 2 batch pipeline across multiple AOIs.

**Request body:**
```json
{
  "aoi_names": ["PS10", "Dholera"],
  "project_id": "PS10_Batch_001",
  "run_stage2": true
}
```

**Response:**
```json
{
  "project_id": "PS10_Batch_001",
  "run_id": "PS10_Batch_001_batch_143012",
  "aois_processed": 2,
  "aois_failed": 0,
  "failed_aois": [],
  "total_change_area_ha": 42.3,
  "total_objects": 87
}
```

---

### `GET /v1/objects`

Return a paginated list of change objects.

**Query parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | string | Filter by run ID |
| `aoi_name` | string | Filter by AOI name |
| `page` | integer | Page number (default: 1) |
| `page_size` | integer | Results per page (default: 50, max: 500) |

**Response:**
```json
{
  "total": 23,
  "page": 1,
  "page_size": 50,
  "objects": [
    {
      "object_id": 1,
      "aoi_name": "Dholera",
      "run_id": "Dholera_20261020_143012",
      "area_px": 312,
      "area_m2": 31200.0,
      "centroid_lat": 22.4201,
      "centroid_lon": 72.1847,
      "semantic_class": "Land_Clearing",
      "confidence": 0.82,
      "ndvi_change": -0.243,
      "ndbi_change": 0.091
    }
  ]
}
```

---

### `GET /v1/statistics`

Return aggregate statistics for a pipeline run.

**Query parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `run_id` | string | Yes | Run identifier |
| `aoi_name` | string | Yes | AOI name |

**Response:**
```json
{
  "aoi_name": "Dholera",
  "run_id": "Dholera_20261020_143012",
  "total_pixels": 4840000,
  "change_pixels": 1847,
  "change_area_ha": 18.47,
  "object_count": 23,
  "class_distribution": {
    "Land_Clearing": 8,
    "Building": 7,
    "Road": 4,
    "Unknown": 4
  }
}
```

---

### `GET /v1/reports/{run_id}`

Return the Stage 1 text report for a run.

**Path parameter:** `run_id` — the pipeline run identifier.

**Response:**
```json
{
  "run_id": "Dholera_20261020_143012",
  "report_type": "stage1",
  "content": "============================================================\nGeoAI Platform — Stage 1 Change Detection Report\n..."
}
```

---

### `GET /health`

API health check.

**Response:**
```json
{
  "status": "ok",
  "service": "GeoAI Platform API"
}
```

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Short error description",
  "detail": "Extended error message with context",
  "run_id": "optional_run_id"
}
```

| HTTP Status | Meaning |
|-------------|---------|
| 400 | Invalid request (AOI not found, bad parameters) |
| 404 | Resource not found (report, run ID) |
| 500 | Internal server error |
