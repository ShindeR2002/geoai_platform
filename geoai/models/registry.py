"""
Model registry for the GeoAI Platform.

Provides a central registry that maps model identifiers to their corresponding
:class:`~geoai.models.base.BaseModel` implementations. The registry enables
the inference pipeline to load any registered model by ID from configuration
without hardcoding model class names in pipeline code.

Adding a new model to the platform requires:
  1. Implementing the BaseModel interface for the new model.
  2. Adding an entry to ``MODEL_REGISTRY`` in this module.
  3. No changes to pipeline, API, or any other module.

Single responsibility: register, discover, and instantiate platform models.

Position in dependency hierarchy: models (depends on models/base,
models/production, models/baselines, core).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Type, Union

from geoai.core.exceptions import ModelError, ModelMetadataError, ModelNotFoundError
from geoai.models.base import BaseModel
from geoai.models.baselines.rf_baseline import RFBaselineModel
from geoai.models.production.rf_enhanced import RFEnhancedModel
from geoai.models.baselines.extra_trees import ExtraTreesModel
from geoai.models.baselines.xgboost import XGBoostModel
from geoai.models.baselines.lightgbm import LightGBMModel
from geoai.models.baselines.catboost import CatBoostModel
from geoai.models.baselines.dl_wrapper import FCEFModel, FCSiamConcModel, FCSiamDiffModel, LightweightSiamCNNModel
from geoai.models.transformers import ChangeFormer, BIT, TinyCD, STANet, SNUNet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

# Maps model_id strings to their BaseModel subclass.
# To add a new model: import its class and add an entry here.
MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "rf_enhanced_v1": RFEnhancedModel,
    "rf_baseline_v1": RFBaselineModel,
    "extra_trees": ExtraTreesModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "catboost": CatBoostModel,
    "fc_ef": FCEFModel,
    "fc_siam_conc": FCSiamConcModel,
    "fc_siam_diff": FCSiamDiffModel,
    "lightweight_siam_cnn": LightweightSiamCNNModel,
    "changeformer": ChangeFormer,
    "bit": BIT,
    "tinycd": TinyCD,
    "stanet": STANet,
    "snunet": SNUNet,
}


# ---------------------------------------------------------------------------
# Registry access
# ---------------------------------------------------------------------------


def get_model_class(model_id: str) -> Type[BaseModel]:
    """Return the BaseModel subclass for a given model identifier.

    Args:
        model_id: Model identifier string matching a key in ``MODEL_REGISTRY``.

    Returns:
        The :class:`~geoai.models.base.BaseModel` subclass (not an instance).

    Raises:
        ModelError: If ``model_id`` is not registered.
    """
    cls = MODEL_REGISTRY.get(model_id)
    if cls is None:
        available = list(MODEL_REGISTRY.keys())
        raise ModelError(
            f"Model '{model_id}' is not registered. "
            f"Available models: {available}."
        )
    return cls


def load_model(
    model_id: str,
    model_path: Union[str, Path],
    metadata_path: Optional[Union[str, Path]] = None,
) -> BaseModel:
    """Instantiate and load a model by its registry ID.

    Creates a new instance of the model class registered under ``model_id``,
    then calls its ``load()`` method with the provided path.

    Args:
        model_id: Identifier string matching a key in ``MODEL_REGISTRY``.
        model_path: Path to the serialised model file.
        metadata_path: Optional path to the model metadata JSON file.

    Returns:
        A loaded :class:`~geoai.models.base.BaseModel` instance ready for
        inference.

    Raises:
        ModelError: If ``model_id`` is not registered.
        ModelNotFoundError: If ``model_path`` does not exist.
        ModelError: If loading fails.
    """
    cls = get_model_class(model_id)
    instance = cls()

    logger.info(
        "Loading model '%s' (%s) from '%s'.",
        model_id,
        cls.__name__,
        model_path,
    )

    # RFEnhancedModel.load() accepts an optional metadata_path argument.
    # Other models may not. Use the base signature unless the class supports it.
    if isinstance(instance, RFEnhancedModel) and metadata_path is not None:
        instance.load(model_path, metadata_path=metadata_path)
    else:
        instance.load(model_path)

    logger.info(
        "Model '%s' loaded successfully — %d features.",
        model_id,
        instance.get_expected_feature_count(),
    )
    return instance


def load_model_from_config(config) -> BaseModel:
    """Load the production model using platform configuration.

    Convenience function that reads model path and metadata path from the
    :class:`~geoai.core.config.ModelConfig` and delegates to
    :func:`load_model`.

    Args:
        config: :class:`~geoai.core.config.PlatformConfig` or
            :class:`~geoai.core.config.ModelConfig` instance.

    Returns:
        Loaded production model.

    Raises:
        ModelError: If loading fails.
    """
    # Accept either the full PlatformConfig or the ModelConfig directly.
    model_cfg = getattr(config, "model", config)

    return load_model(
        model_id=model_cfg.model_id,
        model_path=model_cfg.model_path,
        metadata_path=model_cfg.metadata_path,
    )


def list_registered_models() -> list:
    """Return a list of all registered model identifiers.

    Returns:
        List of model ID strings.
    """
    return list(MODEL_REGISTRY.keys())


def is_model_registered(model_id: str) -> bool:
    """Return True if a model identifier is present in the registry.

    Args:
        model_id: Model identifier to check.

    Returns:
        True if registered; False otherwise.
    """
    return model_id in MODEL_REGISTRY
