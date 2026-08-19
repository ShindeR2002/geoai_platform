"""
Platform exception hierarchy for the GeoAI Platform.

Defines all typed exceptions raised throughout the platform. Every exception
carries contextual information about what failed and why, enabling precise
error handling at pipeline boundaries without catching generic exceptions.

Position in dependency hierarchy: core (no internal imports).
"""


class GeoAIPlatformError(Exception):
    """Base exception for all GeoAI Platform errors."""


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigurationError(GeoAIPlatformError):
    """Raised when a configuration file is missing, malformed, or contains
    invalid values."""


class MissingConfigKeyError(ConfigurationError):
    """Raised when a required configuration key is absent.

    Args:
        key: The missing configuration key path (e.g. 'processing.min_object_size_px').
        config_file: The YAML file where the key was expected.
    """

    def __init__(self, key: str, config_file: str) -> None:
        super().__init__(
            f"Required configuration key '{key}' not found in '{config_file}'."
        )
        self.key = key
        self.config_file = config_file


# ---------------------------------------------------------------------------
# I/O and data errors
# ---------------------------------------------------------------------------


class RasterIOError(GeoAIPlatformError):
    """Raised when a raster file cannot be opened, read, or written."""


class RasterNotFoundError(RasterIOError):
    """Raised when a required raster file does not exist at the specified path.

    Args:
        path: The file path that was not found.
    """

    def __init__(self, path: str) -> None:
        super().__init__(f"Raster file not found: '{path}'.")
        self.path = path


class RasterWriteError(RasterIOError):
    """Raised when a raster file cannot be written to the output path.

    Args:
        path: The file path that could not be written.
        reason: A description of the write failure.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Failed to write raster to '{path}': {reason}.")
        self.path = path
        self.reason = reason


# ---------------------------------------------------------------------------
# Raster validation errors
# ---------------------------------------------------------------------------


class RasterValidationError(GeoAIPlatformError):
    """Raised when a raster or raster pair fails a compatibility check."""


class CRSMismatchError(RasterValidationError):
    """Raised when two rasters do not share the same coordinate reference system.

    Args:
        name_a: Identifier of the first raster.
        crs_a: CRS of the first raster.
        name_b: Identifier of the second raster.
        crs_b: CRS of the second raster.
    """

    def __init__(self, name_a: str, crs_a: str, name_b: str, crs_b: str) -> None:
        super().__init__(
            f"CRS mismatch between '{name_a}' ({crs_a}) and '{name_b}' ({crs_b})."
        )
        self.name_a = name_a
        self.crs_a = crs_a
        self.name_b = name_b
        self.crs_b = crs_b


class SpatialDimensionMismatchError(RasterValidationError):
    """Raised when two rasters do not share the same spatial dimensions.

    Args:
        name_a: Identifier of the first raster.
        shape_a: (height, width) of the first raster.
        name_b: Identifier of the second raster.
        shape_b: (height, width) of the second raster.
    """

    def __init__(
        self,
        name_a: str,
        shape_a: tuple,
        name_b: str,
        shape_b: tuple,
    ) -> None:
        super().__init__(
            f"Spatial dimension mismatch: '{name_a}' is {shape_a} but "
            f"'{name_b}' is {shape_b}."
        )
        self.name_a = name_a
        self.shape_a = shape_a
        self.name_b = name_b
        self.shape_b = shape_b


class TransformMismatchError(RasterValidationError):
    """Raised when two rasters do not share the same affine transform.

    Args:
        name_a: Identifier of the first raster.
        name_b: Identifier of the second raster.
    """

    def __init__(self, name_a: str, name_b: str) -> None:
        super().__init__(
            f"Affine transform mismatch between '{name_a}' and '{name_b}'. "
            "Rasters must be spatially co-registered before processing."
        )
        self.name_a = name_a
        self.name_b = name_b


# ---------------------------------------------------------------------------
# Feature engineering errors
# ---------------------------------------------------------------------------


class FeatureEngineeringError(GeoAIPlatformError):
    """Raised when the feature cube cannot be constructed."""


class FeatureCountMismatchError(FeatureEngineeringError):
    """Raised when the assembled feature cube has an unexpected number of features.

    Args:
        expected: Expected number of features.
        actual: Actual number of features found.
    """

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(
            f"Feature count mismatch: expected {expected} features, got {actual}. "
            "Verify that all source rasters were loaded successfully and that "
            "the feature assembly order matches the model's training configuration."
        )
        self.expected = expected
        self.actual = actual


class EmptyDatasetError(FeatureEngineeringError):
    """Raised when the valid pixel mask produces an empty dataset.

    Args:
        total_pixels: Total number of pixels before masking.
        reason: Description of why all pixels were masked.
    """

    def __init__(self, total_pixels: int, reason: str) -> None:
        super().__init__(
            f"Valid pixel mask produced an empty dataset from {total_pixels} total "
            f"pixels. Reason: {reason}."
        )
        self.total_pixels = total_pixels
        self.reason = reason


# ---------------------------------------------------------------------------
# Model errors
# ---------------------------------------------------------------------------


class ModelError(GeoAIPlatformError):
    """Raised when a model operation fails."""


class ModelNotFoundError(ModelError):
    """Raised when a model file does not exist at the expected path.

    Args:
        path: The path where the model was expected.
    """

    def __init__(self, path: str) -> None:
        super().__init__(f"Model file not found: '{path}'.")
        self.path = path


class ModelMetadataError(ModelError):
    """Raised when a model metadata JSON file is missing or invalid.

    Args:
        path: The metadata file path.
        reason: Description of the metadata problem.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Model metadata error at '{path}': {reason}.")
        self.path = path
        self.reason = reason


class InferenceError(ModelError):
    """Raised when model inference fails.

    Args:
        reason: Description of the inference failure.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"Model inference failed: {reason}.")
        self.reason = reason


# ---------------------------------------------------------------------------
# Export errors
# ---------------------------------------------------------------------------


class ExportError(GeoAIPlatformError):
    """Raised when an export operation fails."""


class GeoTIFFExportError(ExportError):
    """Raised when GeoTIFF export fails.

    Args:
        path: The intended output path.
        reason: Description of the failure.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"GeoTIFF export to '{path}' failed: {reason}.")
        self.path = path
        self.reason = reason


class ShapefileExportError(ExportError):
    """Raised when shapefile export fails.

    Args:
        path: The intended output path.
        reason: Description of the failure.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Shapefile export to '{path}' failed: {reason}.")
        self.path = path
        self.reason = reason


# ---------------------------------------------------------------------------
# Pipeline errors
# ---------------------------------------------------------------------------


class PipelineError(GeoAIPlatformError):
    """Raised when a pipeline stage fails."""


class StageExecutionError(PipelineError):
    """Raised when a pipeline stage encounters an unrecoverable error.

    Args:
        stage: The stage name (e.g. 'stage1', 'stage2').
        aoi_name: The AOI being processed when the error occurred.
        reason: Description of the failure.
    """

    def __init__(self, stage: str, aoi_name: str, reason: str) -> None:
        super().__init__(
            f"Stage '{stage}' failed for AOI '{aoi_name}': {reason}."
        )
        self.stage = stage
        self.aoi_name = aoi_name
        self.reason = reason


# ---------------------------------------------------------------------------
# Sensor errors
# ---------------------------------------------------------------------------


class SensorError(GeoAIPlatformError):
    """Raised when a sensor operation or lookup fails."""


class UnknownSensorError(SensorError):
    """Raised when a sensor identifier is not found in the sensor registry.

    Args:
        sensor_id: The unrecognised sensor identifier.
    """

    def __init__(self, sensor_id: str) -> None:
        super().__init__(
            f"Sensor '{sensor_id}' is not registered in the sensor registry. "
            "Add the sensor definition to geoai/preprocessing/registry.py."
        )
        self.sensor_id = sensor_id
