from typing import Dict
from geoai.datasets.dataset_metadata import DatasetMetadata

DATASET_CATALOG: Dict[str, DatasetMetadata] = {
    "dholera_sentinel_v1": DatasetMetadata(
        dataset_id="dholera_sentinel_v1",
        aoi_name="Dholera",
        sensor_eo="sentinel2",
        sensor_sar="sentinel1",
        t1_year=2021,
        t2_year=2024,
        spatial_resolution_m=10.0,
        crs="EPSG:4326",
        preprocessing_status="Preprocessed with valid pixel mask",
        feature_availability=["Red", "Green", "Blue", "SAR", "NDVI", "NDBI", "NDWI", "Delta_NDVI", "Delta_NDBI", "Delta_NDWI", "Delta_SAR"],
        train_test_split_metadata={"test_size": 0.20, "random_state": 42},
        licensing="Proprietary / Research",
        checksum_md5="dholera_raw_checksum_placeholder_v1",
        version="1.0.0",
        
        # Milestone 9 metadata
        country="India",
        region="Gujarat",
        climate_zone="Semi-arid",
        land_cover_composition={"barren": 0.60, "agriculture": 0.25, "urban": 0.15},
        sensor_acquisition_dates={"T1": ["2021-03-15"], "T2": ["2024-03-20"]},
        cloud_cover=0.02,
        season="Dry / Pre-monsoon",
        ground_truth_source="Manual digitization of high-res imagery",
        reference_lat=22.4167,
        reference_lon=72.1833,
        verification_status={
            "country": "verified",
            "region": "verified",
            "climate_zone": "estimated",
            "land_cover_composition": "estimated",
            "sensor_acquisition_dates": "estimated",
            "cloud_cover": "estimated",
            "season": "estimated",
            "ground_truth_source": "verified",
            "reference_lat": "verified",
            "reference_lon": "verified"
        }
    ),
    "ps10_sentinel_v1": DatasetMetadata(
        dataset_id="ps10_sentinel_v1",
        aoi_name="PS10",
        sensor_eo="sentinel2",
        sensor_sar="sentinel1",
        t1_year=2021,
        t2_year=2024,
        spatial_resolution_m=10.0,
        crs="EPSG:4326",
        preprocessing_status="Preprocessed with valid pixel mask",
        feature_availability=["Red", "Green", "Blue", "SAR", "NDVI", "NDBI", "NDWI", "Delta_NDVI", "Delta_NDBI", "Delta_NDWI", "Delta_SAR"],
        train_test_split_metadata={"test_size": 0.20, "random_state": 42},
        licensing="Proprietary / Research",
        checksum_md5="ps10_raw_checksum_placeholder_v1",
        version="1.0.0",
        
        # Milestone 9 metadata
        country="India",
        region="Pinjore-Haryana",
        climate_zone="Subtropical monsoon",
        land_cover_composition={"urban": 0.50, "vegetation": 0.30, "barren": 0.20},
        sensor_acquisition_dates={"T1": ["2021-10-10"], "T2": ["2024-10-12"]},
        cloud_cover=0.05,
        season="Post-monsoon",
        ground_truth_source="Manual digitization of high-res imagery",
        reference_lat=None,
        reference_lon=None,
        verification_status={
            "country": "verified",
            "region": "estimated",
            "climate_zone": "estimated",
            "land_cover_composition": "estimated",
            "sensor_acquisition_dates": "estimated",
            "cloud_cover": "estimated",
            "season": "estimated",
            "ground_truth_source": "verified",
            "reference_lat": "unknown",
            "reference_lon": "unknown"
        }
    )
}
