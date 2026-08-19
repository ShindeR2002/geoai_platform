"""
Platform-wide constants and enumerations for the GeoAI Platform.

Provides named constants, enumerations, and string literals that are shared
across multiple platform modules. All values that carry scientific or domain
meaning and appear in more than one location must be defined here rather than
repeated as literals.

Single responsibility: define constants and enumerations only.

Position in dependency hierarchy: utils (no internal imports — imported by
all layers).
"""

from enum import Enum, unique


# ---------------------------------------------------------------------------
# Feature engineering constants
# ---------------------------------------------------------------------------

# Canonical number of features in the production feature cube.
# Frozen by rf_enhanced.pkl. Must not be changed without model retraining.
CANONICAL_FEATURE_COUNT: int = 18

# Canonical ordered list of feature names.
# Index position corresponds to the column index in the feature matrix.
# This order is a hard scientific constraint (see Implementation Contract §2.2 F-02).
CANONICAL_FEATURE_NAMES: tuple = (
    "Red_2021",    # index 0
    "Green_2021",  # index 1
    "Blue_2021",   # index 2
    "SAR_2021",    # index 3
    "NDVI_2021",   # index 4
    "NDBI_2021",   # index 5
    "NDWI_2021",   # index 6
    "Red_2024",    # index 7
    "Green_2024",  # index 8
    "Blue_2024",   # index 9
    "SAR_2024",    # index 10
    "NDVI_2024",   # index 11
    "NDBI_2024",   # index 12
    "NDWI_2024",   # index 13
    "Delta_NDVI",  # index 14
    "Delta_NDBI",  # index 15
    "Delta_NDWI",  # index 16
    "Delta_SAR",   # index 17
)

# Number of per-epoch features (T1 features = T2 features = 7).
FEATURES_PER_EPOCH: int = 7

# Number of temporal delta features.
DELTA_FEATURE_COUNT: int = 4

# Default T1 epoch year. Configurable; this is the Version 1 default.
DEFAULT_T1_YEAR: int = 2021

# Default T2 epoch year. Configurable; this is the Version 1 default.
DEFAULT_T2_YEAR: int = 2024


# ---------------------------------------------------------------------------
# Object analysis constants
# ---------------------------------------------------------------------------

# Default minimum area in pixels for a connected component to be retained
# as a significant change object. Source: NB06 Cell 8, NB09 Cell 18.
# Exposed in configuration as processing.min_object_size_px.
DEFAULT_MIN_OBJECT_SIZE_PX: int = 50

# Connectivity for connected component labelling.
# Value of 2 corresponds to 8-connectivity (diagonal neighbours included).
# Source: NB06 Cell 3, NB09 Cell 17. Must not be changed.
OBJECT_CONNECTIVITY: int = 2


# ---------------------------------------------------------------------------
# Pseudo-label constants
# ---------------------------------------------------------------------------

# Default change score weights. Source: NB01 Cell 45.
# Exposed in configuration as processing.pseudo_labels.weights.*.
DEFAULT_WEIGHT_SAR: float = 0.35
DEFAULT_WEIGHT_NDVI: float = 0.35
DEFAULT_WEIGHT_NDBI: float = 0.20
DEFAULT_WEIGHT_NDWI: float = 0.10

# Default threshold sigma multiplier. Source: NB01 Cell 51.
# Threshold = mean(change_score) + DEFAULT_THRESHOLD_SIGMA * std(change_score).
DEFAULT_THRESHOLD_SIGMA: float = 1.5


# ---------------------------------------------------------------------------
# Visualisation constants
# ---------------------------------------------------------------------------

# Default contrast stretch percentiles. Source: NB01 Cell 5.
DEFAULT_CONTRAST_STRETCH_LOW_PCT: float = 2.0
DEFAULT_CONTRAST_STRETCH_HIGH_PCT: float = 98.0


# ---------------------------------------------------------------------------
# PS10 challenge constants
# ---------------------------------------------------------------------------

# PS10 submission filename prefix.
PS10_FILENAME_PREFIX: str = "Change_Mask"

# PS10 submission zip naming convention: PS10_{DD-MMM-YYYY}_{TeamName}.zip
PS10_ZIP_PREFIX: str = "PS10"

# Pixel value for change in the binary output mask.
CHANGE_PIXEL_VALUE: int = 1

# Pixel value for no-change in the binary output mask.
NO_CHANGE_PIXEL_VALUE: int = 0


# ---------------------------------------------------------------------------
# Sensor identifiers
# ---------------------------------------------------------------------------

@unique
class SensorID(str, Enum):
    """Canonical sensor identifiers used in the sensor registry.

    String enum so that sensor IDs can be compared directly with string
    values from YAML configuration without explicit conversion.
    """

    SENTINEL2 = "sentinel2"
    SENTINEL1 = "sentinel1"
    LISS4 = "liss4"


# ---------------------------------------------------------------------------
# Index mode identifiers
# ---------------------------------------------------------------------------

@unique
class IndexMode(str, Enum):
    """Spectral index sourcing mode.

    LOAD: Load precomputed index rasters from file (default production path,
          preserves Version 1 behaviour).
    COMPUTE: Derive indices from raw multi-band imagery using spectral equations.
    """

    LOAD = "load"
    COMPUTE = "compute"


# ---------------------------------------------------------------------------
# Stage 2 semantic class constants
# ---------------------------------------------------------------------------

@unique
class ChangeClass(str, Enum):
    """PS10 Stage 2 semantic change object classes.

    Defined in the PS10 problem statement, Section 4 (Stage 2 dataset table).
    Used for object classification, attribute schemas, and export labelling.
    """

    KACHA_TRACK = "Kacha_Track"
    BUILDING = "Building"
    ROAD = "Road"
    LAND_CLEARING = "Land_Clearing"
    LARGE_INFRA = "Large_Infrastructure"
    NEW_SETTLEMENT = "New_Settlement"
    UNKNOWN = "Unknown"


# Numeric class IDs for raster encoding of classified change maps.
# Pixel value corresponds to the class in the classified GeoTIFF.
CLASS_ID_MAP: dict = {
    ChangeClass.UNKNOWN: 0,
    ChangeClass.KACHA_TRACK: 1,
    ChangeClass.BUILDING: 2,
    ChangeClass.ROAD: 3,
    ChangeClass.LAND_CLEARING: 4,
    ChangeClass.LARGE_INFRA: 5,
    ChangeClass.NEW_SETTLEMENT: 6,
}

# Reverse mapping: class ID integer → ChangeClass enum.
ID_CLASS_MAP: dict = {v: k for k, v in CLASS_ID_MAP.items()}


# ---------------------------------------------------------------------------
# Pipeline stage identifiers
# ---------------------------------------------------------------------------

@unique
class PipelineStage(str, Enum):
    """Pipeline stage identifiers used for logging and project management."""

    STAGE1 = "stage1"
    STAGE2 = "stage2"
    STAGE3 = "stage3"
    ALL = "all"
