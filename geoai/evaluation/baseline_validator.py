import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

logger = logging.getLogger(__name__)

def validate_v1_v2_consistency(
    v1_dir: Path,
    v2_dir: Path,
    aoi_name: str
) -> Dict[str, Any]:
    """
    Compare a V1 reference run against a V2 candidate run for scientific consistency.
    """
    v1_dir = Path(v1_dir)
    v2_dir = Path(v2_dir)
    
    results = {
        "aoi_name": aoi_name,
        "v1_path": str(v1_dir),
        "v2_path": str(v2_dir),
        "checks": {},
        "deviations": [],
        "passed_all": True
    }
    
    # 1. Check directories presence
    if not v1_dir.exists():
        results["passed_all"] = False
        results["deviations"].append(f"V1 reference directory '{v1_dir}' does not exist.")
        return results
    if not v2_dir.exists():
        results["passed_all"] = False
        results["deviations"].append(f"V2 candidate directory '{v2_dir}' does not exist.")
        return results

    # 2. Check prediction mask agreement
    v1_tifs = list((v1_dir / "exports").glob("Change_Mask_*.tif")) + list(v1_dir.glob("*.tif"))
    v2_tifs = list((v2_dir / "exports").glob("Change_Mask_*.tif")) + list(v2_dir.glob("*.tif"))
    
    pixel_agreement = 0.0
    shapes_match = False
    
    if v1_tifs and v2_tifs:
        try:
            with rasterio.open(v1_tifs[0]) as src1, rasterio.open(v2_tifs[0]) as src2:
                mask1 = src1.read(1)
                mask2 = src2.read(1)
                
                if mask1.shape == mask2.shape:
                    shapes_match = True
                    # Clean NaNs
                    valid1 = (mask1 == 1) | (mask1 == 0)
                    valid2 = (mask2 == 1) | (mask2 == 0)
                    
                    inter = ((mask1 == 1) & (mask2 == 1)).sum()
                    union = ((mask1 == 1) | (mask2 == 1)).sum()
                    pixel_agreement = float(inter / union) if union > 0 else 1.0
                else:
                    results["deviations"].append(f"Prediction raster shapes mismatch: V1={mask1.shape}, V2={mask2.shape}")
                    results["passed_all"] = False
        except Exception as e:
            results["deviations"].append(f"Error comparing prediction GeoTIFFs: {e}")
            results["passed_all"] = False
    else:
        results["deviations"].append("Missing prediction GeoTIFF mask files in V1 or V2 folders.")
        results["passed_all"] = False

    results["checks"]["prediction_raster"] = {
        "shapes_match": shapes_match,
        "pixel_agreement_iou": round(pixel_agreement, 6)
    }

    # 3. Check shapefile objects agreement
    v1_shps = list((v1_dir / "exports").glob("Change_Mask_*.shp")) + list(v1_dir.glob("*.shp"))
    v2_shps = list((v2_dir / "exports").glob("Change_Mask_*.shp")) + list(v2_dir.glob("*.shp"))
    
    objects_match = False
    v1_count = 0
    v2_count = 0
    
    if v1_shps and v2_shps:
        try:
            gdf1 = gpd.read_file(v1_shps[0])
            gdf2 = gpd.read_file(v2_shps[0])
            
            v1_count = len(gdf1)
            v2_count = len(gdf2)
            
            if v1_count == v2_count:
                objects_match = True
            else:
                results["deviations"].append(f"Objects shapefile feature count mismatch: V1={v1_count}, V2={v2_count}")
                results["passed_all"] = False
        except Exception as e:
            results["deviations"].append(f"Error comparing shapefiles: {e}")
            results["passed_all"] = False
    else:
        # Check GeoJSON fallback
        v1_gjs = list((v1_dir / "exports").glob("*.geojson")) + list(v1_dir.glob("*.geojson"))
        v2_gjs = list((v2_dir / "exports").glob("*.geojson")) + list(v2_dir.glob("*.geojson"))
        if v1_gjs and v2_gjs:
            try:
                gdf1 = gpd.read_file(v1_gjs[0])
                gdf2 = gpd.read_file(v2_gjs[0])
                v1_count = len(gdf1)
                v2_count = len(gdf2)
                if v1_count == v2_count:
                    objects_match = True
                else:
                    results["deviations"].append(f"Objects GeoJSON feature count mismatch: V1={v1_count}, V2={v2_count}")
                    results["passed_all"] = False
            except Exception as e:
                results["deviations"].append(f"Error comparing GeoJSON files: {e}")
                results["passed_all"] = False
        else:
            results["deviations"].append("Missing vector shapefile/GeoJSON files in V1 or V2 folders.")
            results["passed_all"] = False

    results["checks"]["vector_objects"] = {
        "v1_count": v1_count,
        "v2_count": v2_count,
        "counts_match": objects_match
    }

    # 4. Check area statistics alignment
    v1_csvs = list((v1_dir / "exports").glob("Object_Statistics_*.csv")) + list(v1_dir.glob("statistics.csv"))
    v2_csvs = list((v2_dir / "exports").glob("Object_Statistics_*.csv")) + list(v2_dir.glob("statistics.csv"))
    
    area_match = False
    v1_area = 0.0
    v2_area = 0.0
    
    if v1_csvs and v2_csvs:
        try:
            df1 = pd.read_csv(v1_csvs[0])
            df2 = pd.read_csv(v2_csvs[0])
            
            # Sum up area
            col1 = "area_m2" if "area_m2" in df1.columns else "area_ha"
            col2 = "area_m2" if "area_m2" in df2.columns else "area_ha"
            
            v1_area = float(df1[col1].sum())
            v2_area = float(df2[col2].sum())
            
            if np.isclose(v1_area, v2_area, rtol=1e-3):
                area_match = True
            else:
                results["deviations"].append(f"Total area estimate mismatch: V1={v1_area:.2f}, V2={v2_area:.2f}")
                results["passed_all"] = False
        except Exception as e:
            results["deviations"].append(f"Error comparing area CSV stats: {e}")
            results["passed_all"] = False
            
    results["checks"]["area_statistics"] = {
        "v1_total_area": round(v1_area, 2),
        "v2_total_area": round(v2_area, 2),
        "areas_match": area_match
    }

    return results
