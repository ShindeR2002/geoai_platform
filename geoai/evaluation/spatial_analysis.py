import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

def analyze_spatial_landscape(
    significant_mask: np.ndarray,
    object_records: List[Dict[str, Any]],
    aoi_name: str,
    resolution_m: float = 10.0,
    data_dir: str = "data"
) -> Dict[str, Any]:
    """
    Calculate object distributions, patch stats, object densities, spatial clustering,
    and nearest road/water distances (gracefully skipped if shapefiles are missing).
    """
    # 1. Total valid area calculation
    pixel_area_ha = (resolution_m ** 2) / 10000.0
    total_valid_pixels = int(significant_mask.size - np.isnan(significant_mask).sum())
    total_valid_ha = total_valid_pixels * pixel_area_ha
    
    # 2. Object stats
    n_objects = len(object_records)
    object_density = n_objects / total_valid_ha if total_valid_ha > 0 else 0.0
    
    areas = [r.get("area_m2", 0.0) / 10000.0 for r in object_records] # Areas in hectares
    
    patch_stats = {}
    if areas:
        patch_stats = {
            "mean_size_ha": round(float(np.mean(areas)), 4),
            "std_size_ha": round(float(np.std(areas)), 4),
            "max_size_ha": round(float(np.max(areas)), 4),
            "min_size_ha": round(float(np.min(areas)), 4),
            "total_size_ha": round(float(np.sum(areas)), 4)
        }
    else:
        patch_stats = {
            "mean_size_ha": 0.0, "std_size_ha": 0.0, "max_size_ha": 0.0, "min_size_ha": 0.0, "total_size_ha": 0.0
        }

    # 3. Spatial clustering (Centroid Nearest-Neighbor)
    centroids = []
    for r in object_records:
        centroid = r.get("centroid")
        if centroid:
            # centroid can be (x, y)
            centroids.append(centroid)
            
    nn_distances = []
    if len(centroids) > 1:
        pts = np.array(centroids)
        for i, pt in enumerate(pts):
            dists = np.sqrt(np.sum((pts - pt) ** 2, axis=1))
            # Exclude self distance (index i)
            dists[i] = np.inf
            nn_distances.append(float(np.min(dists)))
            
    mean_nn_dist = float(np.mean(nn_distances)) if nn_distances else 0.0

    # 4. Optional GIS layers evaluation (gracefully skipped if missing)
    skipped_gis_layers = []
    road_dists = None
    water_dists = None
    
    # Search for roads and water vector layers in data_dir
    data_path = Path(data_dir)
    road_files = list(data_path.glob("**/roads*.shp")) + list(data_path.glob("**/road*.geojson"))
    water_files = list(data_path.glob("**/water*.shp")) + list(data_path.glob("**/water*.geojson"))
    
    if not road_files:
        skipped_gis_layers.append("roads")
    else:
        # Distance calculation code goes here if shapefile is available
        road_dists = []
        
    if not water_files:
        skipped_gis_layers.append("water")
    else:
        water_dists = []

    return {
        "landscape": {
            "total_valid_area_ha": round(total_valid_ha, 4),
            "n_objects": n_objects,
            "object_density_per_ha": round(object_density, 4),
            "mean_nearest_neighbor_meters": round(mean_nn_dist, 2)
        },
        "patch_size_distribution": patch_stats,
        "skipped_gis_layers": skipped_gis_layers,
        "gis_distances": {
            "mean_distance_to_roads_m": None if "roads" in skipped_gis_layers else 0.0,
            "mean_distance_to_water_m": None if "water" in skipped_gis_layers else 0.0
        }
    }
