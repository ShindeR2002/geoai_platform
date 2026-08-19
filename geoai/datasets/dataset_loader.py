import numpy as np
import logging
from geoai.core.config import load_platform_config
from geoai.datasets.dataset import Dataset
from geoai.datasets.dataset_metadata import DatasetMetadata

logger = logging.getLogger(__name__)

def load_dataset_from_catalog(
    metadata: DatasetMetadata,
    configs_dir: str = "configs",
    preprocessing_config: dict = None,
    features_config: dict = None,
    campaign_type: str = "preprocessing",
    track: str = "production",
) -> Dataset:
    """Load rasters, compute feature cube and pseudo labels, and return a clean Dataset instance."""
    from geoai.pipeline.stage1 import _load_all_rasters
    from geoai.features.feature_cube import build_feature_cube
    from geoai.features.temporal import compute_all_deltas
    from geoai.features.pseudo_labels import generate_pseudo_labels
    from geoai.features.dataset import prepare_training_dataset

    platform_config = load_platform_config(configs_dir)
    aoi = platform_config.get_aoi(metadata.aoi_name)

    # Re-use existing loading pipelines
    rasters = _load_all_rasters(aoi)
    red_t1, green_t1, blue_t1, ndvi_t1, ndbi_t1, ndwi_t1 = rasters[0:6]
    red_t2, green_t2, blue_t2, ndvi_t2, ndbi_t2, ndwi_t2 = rasters[6:12]
    sar_t1, sar_t2 = rasters[12:14]

    # Campaign A: Apply Preprocessing if enabled
    if campaign_type == "preprocessing" and preprocessing_config:
        s1_config = preprocessing_config.get("s1", {})
        if s1_config.get("speckle_filter") == "refined_lee":
            from geoai.preprocessing.pipeline import refined_lee_filter
            window_size = s1_config.get("window_size", 7)
            logger.info("Applying Refined Lee Speckle Filter on SAR VV channels (window_size=%d)", window_size)
            sar_t1 = refined_lee_filter(sar_t1, size=window_size)
            sar_t2 = refined_lee_filter(sar_t2, size=window_size)

    # Build the canonical 18-feature baseline cube
    feature_cube = build_feature_cube(
        red_t1=red_t1, green_t1=green_t1, blue_t1=blue_t1,
        sar_t1=sar_t1, ndvi_t1=ndvi_t1, ndbi_t1=ndbi_t1, ndwi_t1=ndwi_t1,
        red_t2=red_t2, green_t2=green_t2, blue_t2=blue_t2,
        sar_t2=sar_t2, ndvi_t2=ndvi_t2, ndbi_t2=ndbi_t2, ndwi_t2=ndwi_t2,
    )

    # Compute additional bands for Campaign B (Feature Engineering)
    extra_bands = []
    if campaign_type == "feature_engineering" and features_config:
        # Track A: Production
        if track == "production" or track == "experimental":
            if features_config.get("local_variance", {}).get("enabled", False):
                from geoai.preprocessing.pipeline import local_variance_filter
                size = features_config.get("local_variance", {}).get("window_size", 3)
                logger.info("Computing NDVI Local Variance texture feature (window_size=%d)", size)
                lv_t1 = local_variance_filter(ndvi_t1, size=size)
                lv_t2 = local_variance_filter(ndvi_t2, size=size)
                extra_bands.extend([lv_t1, lv_t2])

        # Track B: Experimental (reconstructed NIR approximations)
        if track == "experimental":
            # SAVI
            if features_config.get("savi", {}).get("enabled", False):
                from geoai.preprocessing.pipeline import reconstruct_nir, compute_savi
                logger.warning("[Track B - Experimental Approximation] Reconstructing NIR band to compute SAVI index.")
                nir_t1 = reconstruct_nir(red_t1, ndvi_t1)
                nir_t2 = reconstruct_nir(red_t2, ndvi_t2)
                savi_t1 = compute_savi(nir_t1, red_t1)
                savi_t2 = compute_savi(nir_t2, red_t2)
                extra_bands.extend([savi_t1, savi_t2])

            # MSAVI
            if features_config.get("msavi", {}).get("enabled", False):
                from geoai.preprocessing.pipeline import reconstruct_nir, compute_msavi
                logger.warning("[Track B - Experimental Approximation] Reconstructing NIR band to compute MSAVI index.")
                nir_t1 = reconstruct_nir(red_t1, ndvi_t1)
                nir_t2 = reconstruct_nir(red_t2, ndvi_t2)
                msavi_t1 = compute_msavi(nir_t1, red_t1)
                msavi_t2 = compute_msavi(nir_t2, red_t2)
                extra_bands.extend([msavi_t1, msavi_t2])

    # Campaign C: Boundary Engineering
    if campaign_type == "boundary_engineering" and features_config:
        boundary_config = features_config.get("boundary", {})
        if boundary_config is None:
            boundary_config = {}
        method = str(boundary_config.get("method", "sobel") or "none").lower()
        
        from geoai.features.boundary import (
            compute_continuous_gradient,
            compute_distance_transform_from_edges,
            compute_morphological_gradient,
            compute_morphological_boundaries,
            project_object_shapes
        )
        
        # Calculate intermediate deltas
        t_delta_ndvi, _, _, t_delta_sar = compute_all_deltas(
            ndvi_t1=ndvi_t1, ndvi_t2=ndvi_t2,
            ndbi_t1=ndbi_t1, ndbi_t2=ndbi_t2,
            ndwi_t1=ndwi_t1, ndwi_t2=ndwi_t2,
            sar_t1=sar_t1, sar_t2=sar_t2
        )
        
        if method in ("sobel", "scharr", "laplacian"):
            f_ndvi = compute_continuous_gradient(t_delta_ndvi, method=method)
            f_sar = compute_continuous_gradient(t_delta_sar, method=method)
            extra_bands.extend([f_ndvi, f_sar])
        elif method == "morphological_gradient":
            f_ndvi = compute_morphological_gradient(t_delta_ndvi)
            f_sar = compute_morphological_gradient(t_delta_sar)
            extra_bands.extend([f_ndvi, f_sar])
        elif method == "distance_transform":
            f_ndvi = compute_distance_transform_from_edges(t_delta_ndvi)
            f_sar = compute_distance_transform_from_edges(t_delta_sar)
            extra_bands.extend([f_ndvi, f_sar])
        elif method == "distance_to_boundary":
            # Distance from baseline pseudo-labels boundary
            t_change_mask, _, _ = generate_pseudo_labels(
                delta_sar=t_delta_sar,
                delta_ndvi=t_delta_ndvi,
                delta_ndbi=ndbi_t2 - ndbi_t1,
                delta_ndwi=ndwi_t2 - ndwi_t1,
                pseudo_label_config=platform_config.processing.pseudo_labels,
            )
            # Find boundary using simple dilation difference
            from scipy.ndimage import binary_dilation
            struct = np.ones((3, 3))
            boundary = binary_dilation(t_change_mask, structure=struct) & ~t_change_mask
            from scipy.ndimage import distance_transform_edt
            f_dist = distance_transform_edt(~boundary)
            extra_bands.append(f_dist)
        elif method == "morphological":
            mask_ndvi = np.abs(t_delta_ndvi) > 0.15
            mask_sar = np.abs(t_delta_sar) > 0.15
            
            i_ndvi, e_ndvi, t_ndvi = compute_morphological_boundaries(mask_ndvi)
            i_sar, e_sar, t_sar = compute_morphological_boundaries(mask_sar)
            
            extra_bands.extend([i_ndvi, i_sar, e_ndvi, e_sar, t_ndvi, t_sar])
        elif method == "boundary_refinement":
            t_change_mask, _, _ = generate_pseudo_labels(
                delta_sar=t_delta_sar,
                delta_ndvi=t_delta_ndvi,
                delta_ndbi=ndbi_t2 - ndbi_t1,
                delta_ndwi=ndwi_t2 - ndwi_t1,
                pseudo_label_config=platform_config.processing.pseudo_labels,
            )
            from scipy.ndimage import label
            label_arr, num_feats = label(t_change_mask)
            # Project shapes using valid mask (ndvi_t1 != 0.0)
            shapes = project_object_shapes(label_arr, ndvi_t1 != 0.0)
            extra_bands.extend([
                shapes["solidity"],
                shapes["compactness"],
                shapes["elongation"],
                shapes["eccentricity"],
                shapes["convexity"]
            ])

    if extra_bands:
        extra_cube = np.stack(extra_bands, axis=-1)
        feature_cube = np.concatenate([feature_cube, extra_cube], axis=-1)

    delta_ndvi, delta_ndbi, delta_ndwi, delta_sar = compute_all_deltas(
        ndvi_t1=ndvi_t1, ndvi_t2=ndvi_t2,
        ndbi_t1=ndbi_t1, ndbi_t2=ndbi_t2,
        ndwi_t1=ndwi_t1, ndwi_t2=ndwi_t2,
        sar_t1=sar_t1, sar_t2=sar_t2
    )

    change_mask, _, _ = generate_pseudo_labels(
        delta_sar=delta_sar,
        delta_ndvi=delta_ndvi,
        delta_ndbi=delta_ndbi,
        delta_ndwi=delta_ndwi,
        pseudo_label_config=platform_config.processing.pseudo_labels,
    )

    X_clean, y_clean, _, _ = prepare_training_dataset(
        feature_cube=feature_cube,
        labels=change_mask
    )

    return Dataset(
        name=metadata.dataset_id,
        X=X_clean,
        y=y_clean,
        metadata=metadata
    )
