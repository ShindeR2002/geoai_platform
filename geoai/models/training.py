"""
Model training for the GeoAI Platform.

Implements the training workflows for the Random Forest Enhanced (production)
and Random Forest Baseline models, reproducing the exact hyperparameters and
data-splitting strategy from Version 1 Notebooks 05 and 04 respectively.

Training is separated from inference so that the platform can retrain models
on new AOIs without modifying the inference pipeline.

Scientific constants (from Version 1 — must not be changed without revalidation):
  RF Enhanced: n_estimators=100, random_state=42, n_jobs=-1, class_weight=None
  RF Baseline: n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
  Train/test split: test_size=0.20, random_state=42, stratify=y

Single responsibility: train sklearn Random Forest models with Version 1
hyperparameters.

Position in dependency hierarchy: models (depends on core, utils, models/base).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from geoai.core.config import ModelConfig
from geoai.core.exceptions import ModelError
from geoai.utils.constants import CANONICAL_FEATURE_COUNT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training constants (Version 1 NB05 / NB04 — frozen by contract)
# ---------------------------------------------------------------------------

_RF_ENHANCED_N_ESTIMATORS: int = 100
_RF_ENHANCED_RANDOM_STATE: int = 42
_RF_ENHANCED_N_JOBS: int = -1
_RF_ENHANCED_CLASS_WEIGHT = None          # NOT 'balanced' — this is intentional

_RF_BASELINE_N_ESTIMATORS: int = 100
_RF_BASELINE_RANDOM_STATE: int = 42
_RF_BASELINE_N_JOBS: int = -1
_RF_BASELINE_CLASS_WEIGHT: str = "balanced"

_TRAIN_TEST_SPLIT_SIZE: float = 0.20
_TRAIN_TEST_RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# RF Enhanced training (production model)
# ---------------------------------------------------------------------------


def train_rf_enhanced(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = _RF_ENHANCED_N_ESTIMATORS,
    random_state: int = _RF_ENHANCED_RANDOM_STATE,
    n_jobs: int = _RF_ENHANCED_N_JOBS,
    test_size: float = _TRAIN_TEST_SPLIT_SIZE,
) -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """Train the Random Forest Enhanced model on an 18-feature dataset.

    Replicates the training procedure from Version 1 Notebook 05 exactly:
    stratified 80/20 train/test split followed by RF training with no
    class weighting.

    Args:
        X: Feature matrix of shape (N, 18). Must have exactly 18 features.
        y: Label vector of shape (N,) with binary labels (0 = no change,
           1 = change).
        n_estimators: Number of trees in the forest. Default 100 (V1 NB05).
        random_state: Random seed for reproducibility. Default 42 (V1 NB05).
        n_jobs: Parallel jobs. -1 uses all CPUs. Default -1 (V1 NB05).
        test_size: Proportion of data held out for evaluation. Default 0.20.

    Returns:
        Tuple of:
            - ``rf``: Trained RandomForestClassifier instance.
            - ``X_test``: Test feature matrix of shape (N_test, 18).
            - ``y_test``: Test label vector of shape (N_test,).

    Raises:
        ModelError: If X does not have exactly 18 features.
        ModelError: If training fails.
    """
    _validate_training_data(X, y, expected_features=CANONICAL_FEATURE_COUNT)

    logger.info(
        "Training RF Enhanced — samples=%d features=%d "
        "n_estimators=%d random_state=%d.",
        X.shape[0], X.shape[1], n_estimators, random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=_TRAIN_TEST_RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Train/test split — train=%d test=%d (%.0f/%.0f).",
        len(X_train), len(X_test),
        100 * (1 - test_size), 100 * test_size,
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        # class_weight is intentionally NOT set (None) for RF Enhanced.
        # This distinguishes it from the baseline which uses 'balanced'.
    )

    try:
        rf.fit(X_train, y_train)
    except Exception as exc:
        raise ModelError(f"RF Enhanced training failed: {exc}.") from exc

    train_score = float(rf.score(X_train, y_train))
    test_score = float(rf.score(X_test, y_test))

    logger.info(
        "RF Enhanced training complete — "
        "train_accuracy=%.4f test_accuracy=%.4f.",
        train_score, test_score,
    )
    return rf, X_test, y_test


def train_rf_enhanced_from_config(
    X: np.ndarray,
    y: np.ndarray,
    model_config: ModelConfig,
) -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """Train RF Enhanced using hyperparameters from the model configuration.

    Args:
        X: Feature matrix of shape (N, 18).
        y: Label vector of shape (N,).
        model_config: :class:`~geoai.core.config.ModelConfig` providing
            n_estimators, random_state, and n_jobs.

    Returns:
        Tuple of (trained_rf, X_test, y_test).
    """
    return train_rf_enhanced(
        X=X,
        y=y,
        n_estimators=model_config.n_estimators,
        random_state=model_config.random_state,
        n_jobs=model_config.n_jobs,
    )


# ---------------------------------------------------------------------------
# RF Baseline training (comparison model)
# ---------------------------------------------------------------------------


def train_rf_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = _RF_BASELINE_N_ESTIMATORS,
    random_state: int = _RF_BASELINE_RANDOM_STATE,
    n_jobs: int = _RF_BASELINE_N_JOBS,
    test_size: float = _TRAIN_TEST_SPLIT_SIZE,
) -> Tuple[RandomForestClassifier, np.ndarray, np.ndarray]:
    """Train the Random Forest Baseline model on a 14-feature dataset.

    Replicates Version 1 Notebook 04 exactly. Uses class_weight='balanced'
    and accepts 14 features (no temporal delta block).

    Args:
        X: Feature matrix of shape (N, 14).
        y: Label vector of shape (N,).
        n_estimators: Number of trees. Default 100 (V1 NB04).
        random_state: Random seed. Default 42 (V1 NB04).
        n_jobs: Parallel jobs. Default -1.
        test_size: Fraction held out for test. Default 0.20.

    Returns:
        Tuple of (trained_rf, X_test, y_test).

    Raises:
        ModelError: If X does not have exactly 14 features or training fails.
    """
    from geoai.models.baselines.rf_baseline import BASELINE_FEATURE_COUNT
    _validate_training_data(X, y, expected_features=BASELINE_FEATURE_COUNT)

    logger.info(
        "Training RF Baseline — samples=%d features=%d class_weight='balanced'.",
        X.shape[0], X.shape[1],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=_TRAIN_TEST_RANDOM_STATE,
        stratify=y,
    )

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=_RF_BASELINE_CLASS_WEIGHT,  # 'balanced' — differs from Enhanced
    )

    try:
        rf.fit(X_train, y_train)
    except Exception as exc:
        raise ModelError(f"RF Baseline training failed: {exc}.") from exc

    test_score = float(rf.score(X_test, y_test))
    logger.info(
        "RF Baseline training complete — test_accuracy=%.4f.", test_score
    )
    return rf, X_test, y_test


# ---------------------------------------------------------------------------
# Model serialisation
# ---------------------------------------------------------------------------


def save_model(
    rf: RandomForestClassifier,
    output_path: Union[str, Path],
) -> Path:
    """Serialise a trained Random Forest model to disk using joblib.

    Args:
        rf: Trained RandomForestClassifier instance.
        output_path: Destination file path. Should end with ``.pkl``.

    Returns:
        Resolved absolute path of the written model file.

    Raises:
        ModelError: If serialisation fails.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving model to '%s'.", output_path)
    try:
        joblib.dump(rf, output_path)
    except Exception as exc:
        raise ModelError(
            f"Failed to serialise model to '{output_path}': {exc}."
        ) from exc

    size_mb = output_path.stat().st_size / (1024 ** 2)
    logger.info(
        "Model saved — '%s' (%.1f MB).", output_path.name, size_mb
    )
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_training_data(
    X: np.ndarray,
    y: np.ndarray,
    expected_features: int,
) -> None:
    """Validate training data shapes and feature count.

    Args:
        X: Feature matrix to validate.
        y: Label vector to validate.
        expected_features: Expected number of columns in X.

    Raises:
        ModelError: If X does not have the expected number of features or
            X and y have incompatible lengths.
    """
    if X.ndim != 2:
        raise ModelError(
            f"Training feature matrix must be 2D (N, {expected_features}), "
            f"got shape {X.shape}."
        )
    if X.shape[1] != expected_features:
        raise ModelError(
            f"Training feature matrix has {X.shape[1]} features but "
            f"{expected_features} are required."
        )
    if len(X) != len(y):
        raise ModelError(
            f"Feature matrix has {len(X)} rows but label vector has {len(y)} "
            "elements. X and y must have the same length."
        )
    if len(X) == 0:
        raise ModelError("Training dataset is empty.")

    unique_classes = np.unique(y)
    logger.debug(
        "Training data validated — samples=%d features=%d classes=%s.",
        X.shape[0], X.shape[1], unique_classes.tolist(),
    )
