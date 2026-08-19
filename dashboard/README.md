# GeoAI Platform Dashboard

This directory contains the Streamlit dashboard for the GeoAI Platform. It provides:

- AOI and pipeline run selection
- Stage 1 & Stage 3 execution
- GeoJSON map inspection
- Object statistics browsing
- Export downloads
- Configuration and model metadata views

## Launching the dashboard

From the repository root:

```bash
streamlit run dashboard/app.py
```

If the environment does not already include the Streamlit dependencies, install them first:

```bash
pip install streamlit folium streamlit-folium
```

## Notes

- The dashboard reads the existing `configs/` directory and `outputs/` run folders.
- A successful Stage 1 run creates the `exports/`, `reports/`, and `figures/` subfolders used by the dashboard.
- The dashboard uses `COMPARISON.json` and `comparison_masks.png` from the `outputs/` root if available.
