"""
ML dataset preparation for the GeoAI Platform.

Converts the spatial (H, W, 18) feature cube into a flat (N_valid, 18) feature
matrix ready for scikit-learn model training and inference. Applies the
canonical compound valid pixel mask from Version 1 Notebook 09 Cell 12 to
exclude pixels with NaN or all-zero features.

The valid pixel mask and its application logic are identical for both training
and inference, ensuring that the same pixel selection criterion is applied
throughout the model lifecycle.

Single responsibility: flatten feature cube, apply valid pixel mask, and
prepare X/y pairs for model training.

Position in dependency hierarchy: features (depends on core, utils).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from geoai.core.exceptions import EmptyDatasetError, FeatureCountMismatchError
from geoai.utils.constants import CANONICAL_FEATURE_COUNT
from geoai.utils.raster_utils import (
    build_valid_pixel_mask,
    flatten_spatial,
    reconstruct_prediction_raster,
    restore_spatial,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def prepare_inference_dataset(
    feature_cube: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Prepare a feature matrix for model inference from a feature cube.

    Flattens the spatial feature cube, applies the canonical valid pixel mask,
    and returns only the valid rows for prediction. The mask and original
    spatial shape are returned so the prediction can be reconstructed into a
    raster afterwards using :func:`reconstruct_spatial_prediction`.

    This function implements the dataset preparation from Version 1 NB09 Cells
    11–12 (the canonical inference path).

    Args:
        feature_cube: Float32 array of shape (H, W, 18) produced by
            ``geoai.features.feature_cube.build_feature_cube``.

    Returns:
        Tuple of three elements:
            - ``X_predict``: float32 array of shape (N_valid, 18) containing
              only the valid pixel feature vectors.
            - ``valid_mask``: bool array of shape (H*W,) marking which pixels
              are valid.
            - ``spatial_shape``: Tuple of (H, W) — the original spatial
              dimensions needed for reconstruction.

    Raises:
        FeatureCountMismatchError: If the cube does not have 18 features.
        EmptyDatasetError: If the valid pixel mask produces an empty dataset.
    """
    _validate_feature_cube(feature_cube)
    H, W, _ = feature_cube.shape

    logger.info(
        "Preparing inference dataset from feature cube of shape %s.", feature_cube.shape
    )

    X_flat = flatten_spatial(feature_cube)
    # Exclude SAR VV (3, 10) and Delta SAR (17) from the zero-mask calculation to ensure
    # identical valid pixel boundaries regardless of speckle filtering operations.
    non_sar_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
    not_nan = ~np.isnan(X_flat).any(axis=1)
    not_all_zero = ~np.all(X_flat[:, non_sar_indices] == 0, axis=1)
    valid_mask = not_nan & not_all_zero

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise EmptyDatasetError(
            total_pixels=H * W,
            reason="All pixels were excluded by the valid pixel mask (all NaN or all zero).",
        )

    X_predict = X_flat[valid_mask]

    logger.info(
        "Inference dataset ready — valid pixels: %d / %d (%.1f%%).",
        n_valid, H * W, 100.0 * n_valid / (H * W),
    )
    return X_predict, valid_mask, (H, W)


def prepare_training_dataset(
    feature_cube: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Prepare feature matrix and label vector for model training.

    Applies the canonical valid pixel mask to both the feature cube and the
    label array, ensuring consistency between features and labels. Rows
    corresponding to invalid pixels are excluded from both X and y.

    Args:
        feature_cube: Float32 array of shape (H, W, 18).
        labels: 2D array of shape (H, W) containing class labels for each
            pixel. Typically the output of pseudo-label generation.

    Returns:
        Tuple of four elements:
            - ``X_clean``: float32 array of shape (N_valid, 18).
            - ``y_clean``: 1D array of shape (N_valid,) with integer labels.
            - ``valid_mask``: bool array of shape (H*W,).
            - ``spatial_shape``: Tuple of (H, W).

    Raises:
        FeatureCountMismatchError: If the cube does not have 18 features.
        EmptyDatasetError: If the valid pixel mask produces an empty dataset.
        ValueError: If feature_cube and labels have incompatible shapes.
    """
    _validate_feature_cube(feature_cube)
    H, W, _ = feature_cube.shape

    if labels.shape != (H, W):
        raise ValueError(
            f"labels shape {labels.shape} does not match feature_cube spatial "
            f"dimensions ({H}, {W})."
        )

    logger.info(
        "Preparing training dataset from feature cube %s and labels %s.",
        feature_cube.shape, labels.shape,
    )

    X_flat = flatten_spatial(feature_cube)
    y_flat = labels.reshape(-1)
    # Exclude SAR VV (3, 10) and Delta SAR (17) from the zero-mask calculation to ensure
    # identical valid pixel boundaries regardless of speckle filtering operations.
    non_sar_indices = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
    not_nan = ~np.isnan(X_flat).any(axis=1)
    not_all_zero = ~np.all(X_flat[:, non_sar_indices] == 0, axis=1)
    valid_mask = not_nan & not_all_zero

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise EmptyDatasetError(
            total_pixels=H * W,
            reason="All pixels were excluded by the valid pixel mask.",
        )

    X_clean = X_flat[valid_mask]
    y_clean = y_flat[valid_mask]

    unique_classes, counts = np.unique(y_clean, return_counts=True)
    class_summary = dict(zip(unique_classes.tolist(), counts.tolist()))

    logger.info(
        "Training dataset ready — valid pixels: %d / %d, "
        "class distribution: %s.",
        n_valid, H * W, class_summary,
    )
    return X_clean, y_clean, valid_mask, (H, W)


# ---------------------------------------------------------------------------
# Prediction reconstruction
# ---------------------------------------------------------------------------


def reconstruct_spatial_prediction(
    predictions: np.ndarray,
    valid_mask: np.ndarray,
    spatial_shape: Tuple[int, int],
) -> np.ndarray:
    """Reconstruct a spatial (H, W) prediction raster from valid-pixel predictions.

    Fills invalid pixel positions with NaN, then reshapes the flat array
    back to the original (H, W) spatial grid. This implements the Version 1
    NB09 Cells 14–15 reconstruction pattern.

    Args:
        predictions: 1D array of shape (N_valid,) from model inference.
        valid_mask: Bool array of shape (H*W,) from :func:`prepare_inference_dataset`.
        spatial_shape: Tuple (H, W) from :func:`prepare_inference_dataset`.

    Returns:
        Float32 prediction raster of shape (H, W) with NaN at invalid positions.

    Raises:
        ValueError: If len(predictions) != valid_mask.sum().
    """
    H, W = spatial_shape
    total_pixels = H * W

    prediction_flat = reconstruct_prediction_raster(
        predictions=predictions.astype(np.float32),
        valid_mask=valid_mask,
        total_pixels=total_pixels,
    )
    prediction_map = restore_spatial(prediction_flat, H, W)

    n_change = int((prediction_map == 1).sum())
    n_nochange = int((prediction_map == 0).sum())
    n_nan = int(np.isnan(prediction_map).sum())

    logger.info(
        "Prediction raster reconstructed — shape=%s change=%d no_change=%d nan=%d.",
        prediction_map.shape, n_change, n_nochange, n_nan,
    )
    return prediction_map


# ---------------------------------------------------------------------------
# Dataset persistence
# ---------------------------------------------------------------------------


def save_dataset(
    X: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    prefix: str = "dataset",
) -> Tuple[Path, Path]:
    """Save feature matrix and label vector to NumPy binary files.

    Args:
        X: Feature matrix of shape (N, 18).
        y: Label vector of shape (N,).
        output_dir: Directory where the .npy files will be written.
        prefix: Filename prefix (e.g. 'X_v2_clean' becomes 'X_v2_clean.npy').

    Returns:
        Tuple of (X_path, y_path) as resolved absolute Path objects.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X_path = out / f"{prefix}_X.npy"
    y_path = out / f"{prefix}_y.npy"
    np.save(X_path, X)
    np.save(y_path, y)
    logger.info(
        "Dataset saved — X: %s %s, y: %s %s.",
        X_path.name, X.shape, y_path.name, y.shape,
    )
    return X_path.resolve(), y_path.resolve()


def load_dataset(
    X_path: Union[str, Path],
    y_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a previously saved feature matrix and label vector.

    Args:
        X_path: Path to the feature matrix .npy file.
        y_path: Path to the label vector .npy file.

    Returns:
        Tuple of (X, y) NumPy arrays.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    X_path = Path(X_path)
    y_path = Path(y_path)
    if not X_path.exists():
        raise FileNotFoundError(f"Feature matrix file not found: '{X_path}'.")
    if not y_path.exists():
        raise FileNotFoundError(f"Label vector file not found: '{y_path}'.")
    X = np.load(X_path)
    y = np.load(y_path)
    logger.info(
        "Dataset loaded — X: %s %s, y: %s %s.",
        X_path.name, X.shape, y_path.name, y.shape,
    )
    return X, y


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_feature_cube(feature_cube: np.ndarray) -> None:
    """Assert that a feature cube has the correct shape and feature count.

    Args:
        feature_cube: Array to validate.

    Raises:
        FeatureEngineeringError: If the array is not 3-dimensional.
        FeatureCountMismatchError: If the feature count is not 18.
    """
    from geoai.core.exceptions import FeatureEngineeringError
    if feature_cube.ndim != 3:
        raise FeatureEngineeringError(
            f"Feature cube must be 3-dimensional (H, W, C), "
            f"got shape {feature_cube.shape}."
        )
    actual = feature_cube.shape[2]
    if actual < CANONICAL_FEATURE_COUNT:
        raise FeatureCountMismatchError(
            expected=CANONICAL_FEATURE_COUNT, actual=actual
        )

