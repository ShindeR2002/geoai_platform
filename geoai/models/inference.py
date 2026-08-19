"""
Model inference for the GeoAI Platform.

Orchestrates the complete inference workflow: loading a model, predicting on
a valid-pixel feature matrix, reconstructing the spatial prediction raster,
and optionally computing prediction probabilities.

This module implements the canonical inference pattern from Version 1 Notebook
09 Cells 14–15 as a reusable function:

    prediction_map = np.full(H * W, np.nan, dtype=np.float32)
    prediction_map[valid_mask] = rf.predict(X_predict)
    prediction_map = prediction_map.reshape(H, W)

The spatial reconstruction is delegated to
``geoai.features.dataset.reconstruct_spatial_prediction`` to keep this module
focused on orchestration.

Single responsibility: orchestrate model loading and prediction for a
feature matrix, returning a spatial prediction raster.

Position in dependency hierarchy: models (depends on models/base,
models/registry, features/dataset, core).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from geoai.core.config import ModelConfig, PlatformConfig
from geoai.core.exceptions import InferenceError, ModelError
from geoai.features.dataset import (
    prepare_inference_dataset,
    reconstruct_spatial_prediction,
)
from geoai.models.base import BaseModel
from geoai.models.registry import load_model, load_model_from_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary inference entry point
# ---------------------------------------------------------------------------


def run_inference(
    model: BaseModel,
    feature_cube: np.ndarray,
    compute_probabilities: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run model inference on a feature cube and reconstruct a spatial prediction raster.

    This is the primary inference function used by all pipeline stages. It
    handles dataset preparation, prediction, and spatial reconstruction in a
    single call.

    Implements the canonical Version 1 NB09 inference workflow:
    1. Flatten (H, W, 18) → (H*W, 18).
    2. Apply canonical valid pixel mask.
    3. Predict on valid pixels only.
    4. Reconstruct (H, W) raster with NaN at invalid positions.

    Args:
        model: A loaded :class:`~geoai.models.base.BaseModel` instance.
        feature_cube: Float32 array of shape (H, W, 18) from
            :func:`~geoai.features.feature_cube.build_feature_cube`.
        compute_probabilities: If True, also compute prediction probabilities
            for the change class (column 1 of predict_proba output). Default
            False — probabilities are only computed when needed to avoid the
            additional cost.

    Returns:
        Tuple of three elements:
            - ``prediction_map``: Float32 array of shape (H, W). Change pixels
              have value 1.0, no-change pixels 0.0, invalid pixels NaN.
            - ``valid_mask``: Bool array of shape (H*W,) — the applied mask.
            - ``probability_map``: Float32 array of shape (H, W) with P(change)
              values at valid pixel positions, NaN elsewhere. None if
              ``compute_probabilities`` is False.

    Raises:
        FeatureCountMismatchError: If the feature cube has the wrong number
            of features.
        EmptyDatasetError: If no valid pixels exist.
        InferenceError: If the model prediction fails.
    """
    H, W = feature_cube.shape[0], feature_cube.shape[1]
    logger.info(
        "Running inference — model='%s' input_shape=%s.",
        model.model_id,
        feature_cube.shape,
    )

    # Step 1 & 2: Flatten and apply valid pixel mask.
    X_predict, valid_mask, spatial_shape = prepare_inference_dataset(feature_cube)
    logger.info("Valid pixels for inference: %d / %d.", X_predict.shape[0], H * W)

    # Dynamic feature subsetting to match model expectations
    capabilities = model.get_capabilities()
    expected_channels = capabilities.expected_input_channels
    if expected_channels and len(expected_channels) < X_predict.shape[1]:
        from geoai.utils.constants import CANONICAL_FEATURE_NAMES
        col_indices = [
            list(CANONICAL_FEATURE_NAMES).index(channel)
            for channel in expected_channels
            if channel in CANONICAL_FEATURE_NAMES
        ]
        if col_indices:
            X_predict = X_predict[:, col_indices]
            logger.info("Subsampled inference dataset to %d columns to match model capabilities.", len(col_indices))

    # Step 3: Predict class labels on valid pixels only.
    predictions = model.predict(X_predict)

    # Step 4: Reconstruct the (H, W) prediction raster.
    prediction_map = reconstruct_spatial_prediction(
        predictions=predictions,
        valid_mask=valid_mask,
        spatial_shape=spatial_shape,
    )

    # Optional: compute change probabilities.
    probability_map = None
    if compute_probabilities:
        logger.info("Computing prediction probabilities.")
        proba = model.predict_proba(X_predict)
        # Column 1 is P(change); column 0 is P(no change).
        change_proba = proba[:, 1]
        probability_map = reconstruct_spatial_prediction(
            predictions=change_proba,
            valid_mask=valid_mask,
            spatial_shape=spatial_shape,
        )
        logger.info(
            "Probability map ready — mean P(change) at valid pixels=%.4f.",
            float(change_proba.mean()),
        )

    logger.info(
        "Inference complete — prediction_map shape=%s change_px=%d.",
        prediction_map.shape,
        int((prediction_map == 1).sum()),
    )
    return prediction_map, valid_mask, probability_map


# ---------------------------------------------------------------------------
# Config-driven inference
# ---------------------------------------------------------------------------


def run_inference_from_config(
    feature_cube: np.ndarray,
    platform_config: PlatformConfig,
    compute_probabilities: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load the production model from configuration and run inference.

    Convenience function for pipeline stages that receive a
    :class:`~geoai.core.config.PlatformConfig` object.

    Args:
        feature_cube: Float32 feature cube of shape (H, W, 18).
        platform_config: Full platform configuration providing model path
            and model ID.
        compute_probabilities: If True, also compute probability maps.

    Returns:
        Tuple of (prediction_map, valid_mask, probability_map).
        See :func:`run_inference` for details.
    """
    logger.info(
        "Loading model '%s' for config-driven inference.",
        platform_config.model.model_id,
    )
    model = load_model_from_config(platform_config)
    return run_inference(
        model=model,
        feature_cube=feature_cube,
        compute_probabilities=compute_probabilities,
    )


# ---------------------------------------------------------------------------
# Prediction summary
# ---------------------------------------------------------------------------


def summarise_prediction(
    prediction_map: np.ndarray,
    resolution_m: float = 10.0,
) -> dict:
    """Compute summary statistics for a binary prediction raster.

    Args:
        prediction_map: Float32 array of shape (H, W) from :func:`run_inference`.
        resolution_m: Ground sampling distance in metres. Used to compute
            area estimates. Default 10.0 m (Sentinel-2 native resolution).

    Returns:
        Dictionary containing:
            - ``total_pixels``: Total number of pixels in the raster.
            - ``valid_pixels``: Pixels with non-NaN predictions.
            - ``change_pixels``: Pixels predicted as change (value == 1).
            - ``no_change_pixels``: Pixels predicted as no-change (value == 0).
            - ``invalid_pixels``: Pixels excluded by the valid pixel mask (NaN).
            - ``change_fraction``: Change pixels / valid pixels.
            - ``change_area_m2``: Estimated area of change in square metres.
            - ``change_area_ha``: Estimated area of change in hectares.
    """
    total = int(prediction_map.size)
    invalid = int(np.isnan(prediction_map).sum())
    valid = total - invalid
    change = int((prediction_map == 1).sum())
    no_change = int((prediction_map == 0).sum())
    change_fraction = change / max(valid, 1)
    pixel_area_m2 = resolution_m ** 2
    change_area_m2 = change * pixel_area_m2
    change_area_ha = change_area_m2 / 10_000.0

    summary = {
        "total_pixels": total,
        "valid_pixels": valid,
        "change_pixels": change,
        "no_change_pixels": no_change,
        "invalid_pixels": invalid,
        "change_fraction": round(change_fraction, 6),
        "change_area_m2": round(change_area_m2, 2),
        "change_area_ha": round(change_area_ha, 4),
    }

    logger.info(
        "Prediction summary — change=%d px (%.2f%%) ≈ %.2f ha.",
        change, 100.0 * change_fraction, change_area_ha,
    )
    return summary
