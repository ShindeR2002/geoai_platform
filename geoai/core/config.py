"""
Configuration loader for the GeoAI Platform.

Loads, validates, and exposes all platform configuration from YAML files in
the ``configs/`` directory. Returns strongly typed dataclass objects so that
all downstream modules receive validated configuration with IDE-accessible
attribute names rather than raw dictionary lookups.

The top-level :func:`load_platform_config` function is the single entry point
for all configuration loading. It merges all YAML files into one
:class:`PlatformConfig` object that is passed through the platform via
dependency injection.

Position in dependency hierarchy: core (depends on exceptions only).
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from geoai.core.exceptions import ConfigurationError, MissingConfigKeyError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AOI configuration
# ---------------------------------------------------------------------------


@dataclass
class FilePatterns:
    """File naming patterns for a single AOI dataset.

    All patterns use ``{aoi}`` and ``{year}`` as placeholders that are
    substituted at runtime with the AOI name and epoch year.
    """

    rgb: str = "{aoi}_RGB_{year}.tif"
    sar_vv: str = "{aoi}_S1_VV_{year}.tif"
    ndvi: str = "{aoi}_NDVI_{year}.tif"
    ndbi: str = "{aoi}_NDBI_{year}.tif"
    ndwi: str = "{aoi}_NDWI_{year}.tif"


@dataclass
class AOIConfig:
    """Configuration for a single Area of Interest.

    Args:
        name: Human-readable AOI identifier (e.g. 'Dholera', 'PS10').
        t1_year: Year of the first (earlier) epoch.
        t2_year: Year of the second (later) epoch.
        data_dir: Directory containing the raw raster files for this AOI.
        sensor_eo: Identifier of the electro-optical sensor (must exist in
            the sensor registry).
        sensor_sar: Identifier of the SAR sensor (must exist in the sensor
            registry). None if SAR data is not available for this AOI.
        crs: Expected coordinate reference system in EPSG notation (e.g.
            'EPSG:32643'). Used for validation only; does not reproject.
        resolution_m: Expected ground sampling distance in metres.
        file_patterns: File naming patterns for this AOI's rasters.
        index_mode: Index sourcing mode. 'load' uses precomputed index rasters
            (default production path). 'compute' derives indices from raw bands.
        reference_lat: Reference latitude for PS10 output filename convention.
            None if not yet determined.
        reference_lon: Reference longitude for PS10 output filename convention.
            None if not yet determined.
    """

    name: str
    t1_year: int
    t2_year: int
    data_dir: str
    sensor_eo: str = "sentinel2"
    sensor_sar: Optional[str] = "sentinel1"
    crs: Optional[str] = None
    resolution_m: float = 10.0
    file_patterns: FilePatterns = field(default_factory=FilePatterns)
    index_mode: str = "load"
    reference_lat: Optional[float] = None
    reference_lon: Optional[float] = None

    def get_raster_path(self, raster_type: str, year: int) -> Path:
        """Construct the full path to a raster file for this AOI and year.

        Args:
            raster_type: One of 'rgb', 'sar_vv', 'ndvi', 'ndbi', 'ndwi'.
            year: Epoch year (e.g. 2021 or 2024).

        Returns:
            Absolute :class:`pathlib.Path` to the raster file.

        Raises:
            ConfigurationError: If ``raster_type`` is not a known pattern key.
        """
        pattern_map = {
            "rgb": self.file_patterns.rgb,
            "sar_vv": self.file_patterns.sar_vv,
            "ndvi": self.file_patterns.ndvi,
            "ndbi": self.file_patterns.ndbi,
            "ndwi": self.file_patterns.ndwi,
        }
        if raster_type not in pattern_map:
            raise ConfigurationError(
                f"Unknown raster type '{raster_type}' for AOI '{self.name}'. "
                f"Valid types: {list(pattern_map.keys())}."
            )
        pattern = pattern_map[raster_type]
        filename = pattern.format(aoi=self.name, year=year)
        return Path(self.data_dir) / filename


# ---------------------------------------------------------------------------
# Processing configuration
# ---------------------------------------------------------------------------


@dataclass
class PseudoLabelWeights:
    """Weights for the composite pseudo-label change score.

    Weights must sum to 1.0. Default values are taken from Version 1
    Notebook 01 Cell 45 and must not be changed without experimental
    revalidation.
    """

    sar: float = 0.35
    ndvi: float = 0.35
    ndbi: float = 0.20
    ndwi: float = 0.10

    def validate(self) -> None:
        """Assert that weights sum to 1.0 within floating-point tolerance.

        Raises:
            ConfigurationError: If weights do not sum to 1.0.
        """
        total = self.sar + self.ndvi + self.ndbi + self.ndwi
        if abs(total - 1.0) > 1e-6:
            raise ConfigurationError(
                f"Pseudo-label weights must sum to 1.0, but sum is {total:.6f}. "
                "Check processing.pseudo_labels.weights in processing.yaml."
            )


@dataclass
class PseudoLabelConfig:
    """Configuration for pseudo-label generation.

    Args:
        weights: Per-feature weights for the composite change score.
        threshold_sigma: Multiplier applied to the standard deviation when
            computing the change score threshold. Default 1.5 (from V1 NB01).
    """

    weights: PseudoLabelWeights = field(default_factory=PseudoLabelWeights)
    threshold_sigma: float = 1.5


@dataclass
class ProcessingConfig:
    """Top-level processing configuration.

    Args:
        min_object_size_px: Minimum area in pixels for a connected component
            to be retained as a significant object. Default 50 (from V1 NB06/NB09).
        contrast_stretch_low_pct: Lower percentile for contrast stretching
            during visualisation. Default 2 (from V1 NB01).
        contrast_stretch_high_pct: Upper percentile for contrast stretching
            during visualisation. Default 98 (from V1 NB01).
        pseudo_labels: Pseudo-label generation parameters.
        aois: List of AOI configurations for batch processing.
    """

    min_object_size_px: int = 50
    contrast_stretch_low_pct: float = 2.0
    contrast_stretch_high_pct: float = 98.0
    pseudo_labels: PseudoLabelConfig = field(default_factory=PseudoLabelConfig)
    aois: List[AOIConfig] = field(default_factory=list)
    split_policy: str = "spatial"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for the machine learning model.

    Args:
        model_id: Identifier matching an entry in the model registry.
        model_path: Path to the serialised model file (.pkl).
        metadata_path: Path to the model metadata JSON file.
        expected_feature_count: Number of features the model expects. Used
            for pre-inference validation. Default 18 (from V1 NB05).
        n_estimators: Random Forest estimator count. Default 100 (from V1 NB05).
        random_state: Random seed for reproducibility. Default 42 (from V1 NB05).
        n_jobs: Number of parallel jobs. -1 uses all available CPUs.
        class_weight: Class weighting strategy. None for RF Enhanced (production).
    """

    model_id: str = "rf_enhanced_v1"
    model_path: str = "models/rf/rf_enhanced.pkl"
    metadata_path: str = "models/rf/rf_enhanced_meta.json"
    expected_feature_count: int = 18
    n_estimators: int = 100
    random_state: int = 42
    n_jobs: int = -1
    class_weight: Optional[str] = None


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------


@dataclass
class FeatureConfig:
    """Feature engineering configuration.

    The feature list and order are frozen by the production model and must
    not be changed without model retraining.
    """

    feature_names: List[str] = field(default_factory=lambda: [
        "Red_2021", "Green_2021", "Blue_2021", "SAR_2021",
        "NDVI_2021", "NDBI_2021", "NDWI_2021",
        "Red_2024", "Green_2024", "Blue_2024", "SAR_2024",
        "NDVI_2024", "NDBI_2024", "NDWI_2024",
        "Delta_NDVI", "Delta_NDBI", "Delta_NDWI", "Delta_SAR",
    ])

    @property
    def feature_count(self) -> int:
        """Return the total number of features in the canonical feature list.

        Returns:
            Integer count of features (18 in the production configuration).
        """
        return len(self.feature_names)


# ---------------------------------------------------------------------------
# Export configuration
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    """Configuration controlling output file generation.

    Args:
        output_base_dir: Root directory for all pipeline outputs.
        geotiff_compress: Compression algorithm for GeoTIFF outputs.
            'lzw' is recommended for binary masks.
        shapefile_encoding: Character encoding for shapefile attribute tables.
        ps10_submission_prefix: Prefix for PS10 submission filenames.
        include_geojson: Whether to export GeoJSON in addition to shapefile.
        include_csv: Whether to export object statistics CSV.
    """

    output_base_dir: str = "outputs"
    geotiff_compress: str = "lzw"
    shapefile_encoding: str = "utf-8"
    ps10_submission_prefix: str = "Change_Mask"
    include_geojson: bool = True
    include_csv: bool = True


# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------


@dataclass
class APIConfig:
    """Configuration for the REST API server.

    Args:
        host: Host address to bind the API server.
        port: Port number for the API server.
        debug: Enable debug mode (must be False in production).
        cors_origins: List of allowed CORS origins.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


# ---------------------------------------------------------------------------
# Dashboard configuration
# ---------------------------------------------------------------------------


@dataclass
class DashboardConfig:
    """Configuration for the dashboard backend integration layer.

    Args:
        project_manifest_dir: Directory where project manifest JSON files
            are stored.
        default_aoi: Name of the AOI displayed by default in the dashboard.
        tile_format: Format for serving raster tiles to the frontend.
    """

    project_manifest_dir: str = "outputs/projects"
    default_aoi: str = ""
    tile_format: str = "COG"


# ---------------------------------------------------------------------------
# Top-level platform configuration
# ---------------------------------------------------------------------------


@dataclass
class PlatformConfig:
    """Unified platform configuration aggregating all subsystem configs.

    This is the single configuration object passed through the platform via
    dependency injection. All pipeline stages, modules, and the API server
    receive an instance of this class.

    Args:
        processing: Processing and AOI configuration.
        model: Machine learning model configuration.
        features: Feature engineering configuration.
        export: Export and output file configuration.
        api: REST API server configuration.
        dashboard: Dashboard backend configuration.
        data_root: Root directory for all data files. All relative paths in
            AOI configurations are resolved against this root.
    """

    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    api: APIConfig = field(default_factory=APIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    data_root: str = "."

    def get_aoi(self, name: str) -> AOIConfig:
        """Return the AOI configuration for a given AOI name.

        Args:
            name: AOI name to look up (e.g. 'Dholera').

        Returns:
            The :class:`AOIConfig` for the specified AOI.

        Raises:
            ConfigurationError: If no AOI with the given name is configured.
        """
        for aoi in self.processing.aois:
            if aoi.name == name:
                return aoi
        available = [a.name for a in self.processing.aois]
        raise ConfigurationError(
            f"AOI '{name}' not found in configuration. "
            f"Available AOIs: {available}."
        )


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its content as a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ConfigurationError: If the file does not exist or cannot be parsed.
    """
    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: '{path}'. "
            "Ensure the configs/ directory is present and complete."
        )
    try:
        with open(path, "r", encoding="utf-8") as fh:
            content = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse YAML configuration file '{path}': {exc}."
        ) from exc
    if content is None:
        return {}
    return content


def _require(data: dict, key: str, config_file: str) -> Any:
    """Extract a required key from a configuration dictionary.

    Args:
        data: The configuration dictionary to extract from.
        key: The key to extract (supports dotted paths like 'model.path').
        config_file: The config file name, used in the error message.

    Returns:
        The value associated with ``key``.

    Raises:
        MissingConfigKeyError: If ``key`` is not present.
    """
    keys = key.split(".")
    current = data
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            raise MissingConfigKeyError(key=key, config_file=config_file)
        current = current[k]
    return current


def _parse_aoi(aoi_dict: dict) -> AOIConfig:
    """Parse a single AOI dictionary into an :class:`AOIConfig` instance.

    Args:
        aoi_dict: Dictionary from the processing YAML ``aois`` list.

    Returns:
        Populated :class:`AOIConfig`.

    Raises:
        ConfigurationError: If a required AOI field is missing.
    """
    try:
        patterns_raw = aoi_dict.get("file_patterns", {})
        patterns = FilePatterns(
            rgb=patterns_raw.get("rgb", "{aoi}_RGB_{year}.tif"),
            sar_vv=patterns_raw.get("sar_vv", "{aoi}_S1_VV_{year}.tif"),
            ndvi=patterns_raw.get("ndvi", "{aoi}_NDVI_{year}.tif"),
            ndbi=patterns_raw.get("ndbi", "{aoi}_NDBI_{year}.tif"),
            ndwi=patterns_raw.get("ndwi", "{aoi}_NDWI_{year}.tif"),
        )
        return AOIConfig(
            name=aoi_dict["name"],
            t1_year=int(aoi_dict["t1_year"]),
            t2_year=int(aoi_dict["t2_year"]),
            data_dir=aoi_dict["data_dir"],
            sensor_eo=aoi_dict.get("sensor_eo", "sentinel2"),
            sensor_sar=aoi_dict.get("sensor_sar", "sentinel1"),
            crs=aoi_dict.get("crs", None),
            resolution_m=float(aoi_dict.get("resolution_m", 10.0)),
            file_patterns=patterns,
            index_mode=aoi_dict.get("index_mode", "load"),
            reference_lat=aoi_dict.get("reference_lat", None),
            reference_lon=aoi_dict.get("reference_lon", None),
        )
    except KeyError as exc:
        raise ConfigurationError(
            f"AOI configuration is missing required field: {exc}. "
            "Each AOI entry must have at minimum: name, t1_year, t2_year, data_dir."
        ) from exc


def _parse_processing(raw: dict) -> ProcessingConfig:
    """Parse the processing configuration block.

    Args:
        raw: Dictionary from processing.yaml.

    Returns:
        Populated :class:`ProcessingConfig`.
    """
    pl_raw = raw.get("pseudo_labels", {})
    w_raw = pl_raw.get("weights", {})
    weights = PseudoLabelWeights(
        sar=float(w_raw.get("sar", 0.35)),
        ndvi=float(w_raw.get("ndvi", 0.35)),
        ndbi=float(w_raw.get("ndbi", 0.20)),
        ndwi=float(w_raw.get("ndwi", 0.10)),
    )
    weights.validate()
    pseudo_labels = PseudoLabelConfig(
        weights=weights,
        threshold_sigma=float(pl_raw.get("threshold_sigma", 1.5)),
    )
    aois = [_parse_aoi(a) for a in raw.get("aois", [])]
    return ProcessingConfig(
        min_object_size_px=int(raw.get("min_object_size_px", 50)),
        contrast_stretch_low_pct=float(raw.get("contrast_stretch_low_pct", 2.0)),
        contrast_stretch_high_pct=float(raw.get("contrast_stretch_high_pct", 98.0)),
        pseudo_labels=pseudo_labels,
        aois=aois,
        split_policy=str(raw.get("split_policy", "spatial")),
    )


def _parse_model(raw: dict) -> ModelConfig:
    """Parse the model configuration block.

    Args:
        raw: Dictionary from model.yaml.

    Returns:
        Populated :class:`ModelConfig`.
    """
    return ModelConfig(
        model_id=raw.get("model_id", "rf_enhanced_v1"),
        model_path=raw.get("model_path", "models/rf/rf_enhanced.pkl"),
        metadata_path=raw.get("metadata_path", "models/rf/rf_enhanced_meta.json"),
        expected_feature_count=int(raw.get("expected_feature_count", 18)),
        n_estimators=int(raw.get("n_estimators", 100)),
        random_state=int(raw.get("random_state", 42)),
        n_jobs=int(raw.get("n_jobs", -1)),
        class_weight=raw.get("class_weight", None),
    )


def _parse_features(raw: dict) -> FeatureConfig:
    """Parse the features configuration block.

    Args:
        raw: Dictionary from features.yaml.

    Returns:
        Populated :class:`FeatureConfig`.
    """
    feature_names = raw.get("feature_names", None)
    if feature_names is not None:
        return FeatureConfig(feature_names=feature_names)
    return FeatureConfig()


def _parse_export(raw: dict) -> ExportConfig:
    """Parse the export configuration block.

    Args:
        raw: Dictionary from export.yaml.

    Returns:
        Populated :class:`ExportConfig`.
    """
    return ExportConfig(
        output_base_dir=raw.get("output_base_dir", "outputs"),
        geotiff_compress=raw.get("geotiff_compress", "lzw"),
        shapefile_encoding=raw.get("shapefile_encoding", "utf-8"),
        ps10_submission_prefix=raw.get("ps10_submission_prefix", "Change_Mask"),
        include_geojson=bool(raw.get("include_geojson", True)),
        include_csv=bool(raw.get("include_csv", True)),
    )


def _parse_api(raw: dict) -> APIConfig:
    """Parse the API configuration block.

    Args:
        raw: Dictionary from api.yaml.

    Returns:
        Populated :class:`APIConfig`.
    """
    return APIConfig(
        host=raw.get("host", "0.0.0.0"),
        port=int(raw.get("port", 8000)),
        debug=bool(raw.get("debug", False)),
        cors_origins=raw.get("cors_origins", ["*"]),
    )


def _parse_dashboard(raw: dict) -> DashboardConfig:
    """Parse the dashboard configuration block.

    Args:
        raw: Dictionary from dashboard.yaml.

    Returns:
        Populated :class:`DashboardConfig`.
    """
    return DashboardConfig(
        project_manifest_dir=raw.get("project_manifest_dir", "outputs/projects"),
        default_aoi=raw.get("default_aoi", ""),
        tile_format=raw.get("tile_format", "COG"),
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_platform_config(configs_dir: str = "configs") -> PlatformConfig:
    """Load and validate the complete platform configuration from YAML files.

    Reads all YAML files in ``configs_dir`` and assembles them into a single
    :class:`PlatformConfig` object. Missing optional YAML files result in
    default values being used. Missing required fields within present files
    raise :class:`ConfigurationError`.

    Args:
        configs_dir: Path to the directory containing the YAML configuration
            files. Defaults to ``'configs'`` relative to the working directory.

    Returns:
        A fully populated and validated :class:`PlatformConfig` instance.

    Raises:
        ConfigurationError: If any required configuration is missing or invalid.
    """
    base = Path(configs_dir)
    logger.info("Loading platform configuration from '%s'.", base)

    def _safe_load(filename: str) -> dict:
        path = base / filename
        if path.exists():
            return _load_yaml(path)
        logger.warning(
            "Configuration file '%s' not found. Using defaults for this section.",
            path,
        )
        return {}

    processing_raw = _safe_load("processing.yaml")
    model_raw = _safe_load("model.yaml")
    features_raw = _safe_load("features.yaml")
    export_raw = _safe_load("export.yaml")
    api_raw = _safe_load("api.yaml")
    dashboard_raw = _safe_load("dashboard.yaml")

    config = PlatformConfig(
        processing=_parse_processing(processing_raw),
        model=_parse_model(model_raw),
        features=_parse_features(features_raw),
        export=_parse_export(export_raw),
        api=_parse_api(api_raw),
        dashboard=_parse_dashboard(dashboard_raw),
        data_root=processing_raw.get("data_root", "."),
    )

    logger.info(
        "Configuration loaded — %d AOI(s) configured, model='%s', "
        "feature_count=%d, output_dir='%s'.",
        len(config.processing.aois),
        config.model.model_id,
        config.features.feature_count,
        config.export.output_base_dir,
    )
    return config
