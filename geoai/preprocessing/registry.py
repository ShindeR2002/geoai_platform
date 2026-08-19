"""
Sensor capability registry for the GeoAI Platform.

Defines the capabilities of each supported remote sensing sensor, including
band-to-wavelength mappings, default resolutions, and the band indices required
for spectral index computation in Mode 2 (compute from raw bands).

Adding support for a new sensor requires only adding an entry to
``SENSOR_REGISTRY`` in this module. No changes to preprocessing, feature
engineering, or pipeline modules are required.

Single responsibility: define and expose sensor capability metadata.

Position in dependency hierarchy: preprocessing (depends on constants, exceptions).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

from geoai.core.exceptions import UnknownSensorError
from geoai.utils.constants import SensorID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sensor definition dataclass
# ---------------------------------------------------------------------------


@dataclass
class SensorSpec:
    """Specification of a remote sensing sensor's capabilities.

    Args:
        sensor_id: Canonical identifier matching a :class:`SensorID` enum value.
        display_name: Human-readable sensor name for logging and reports.
        resolution_m: Native ground sampling distance in metres.
        rgb_band_indices: 1-indexed band numbers for Red, Green, Blue
            (in that order). None if the sensor has no RGB bands.
        ndvi_bands: Tuple of 1-indexed (NIR, Red) band numbers for NDVI
            computation. None if NDVI cannot be computed.
        ndbi_bands: Tuple of 1-indexed (SWIR, NIR) band numbers for NDBI
            computation. None if NDBI cannot be computed.
        ndwi_bands: Tuple of 1-indexed (Green, NIR) band numbers for NDWI
            computation. None if NDWI cannot be computed.
        vv_band_index: 1-indexed band number for SAR VV polarisation.
            None if the sensor is not a SAR system.
        is_eo: True if this is an electro-optical sensor.
        is_sar: True if this is a Synthetic Aperture Radar sensor.
        notes: Additional notes about the sensor or its band configuration.
    """

    sensor_id: str
    display_name: str
    resolution_m: float
    rgb_band_indices: Optional[Tuple[int, int, int]] = None
    ndvi_bands: Optional[Tuple[int, int]] = None
    ndbi_bands: Optional[Tuple[int, int]] = None
    ndwi_bands: Optional[Tuple[int, int]] = None
    vv_band_index: Optional[int] = None
    is_eo: bool = True
    is_sar: bool = False
    notes: list = field(default_factory=list)

    def can_compute_ndvi(self) -> bool:
        """Return True if this sensor has the bands required for NDVI computation.

        Returns:
            True if ``ndvi_bands`` is not None.
        """
        return self.ndvi_bands is not None

    def can_compute_ndbi(self) -> bool:
        """Return True if this sensor has the bands required for NDBI computation.

        Returns:
            True if ``ndbi_bands`` is not None.
        """
        return self.ndbi_bands is not None

    def can_compute_ndwi(self) -> bool:
        """Return True if this sensor has the bands required for NDWI computation.

        Returns:
            True if ``ndwi_bands`` is not None.
        """
        return self.ndwi_bands is not None

    def can_compute_all_indices(self) -> bool:
        """Return True if this sensor supports all three required spectral indices.

        Returns:
            True if NDVI, NDBI, and NDWI can all be computed.
        """
        return (
            self.can_compute_ndvi()
            and self.can_compute_ndbi()
            and self.can_compute_ndwi()
        )


# ---------------------------------------------------------------------------
# Sensor registry
# ---------------------------------------------------------------------------

# The sensor registry maps sensor ID strings to SensorSpec instances.
# Adding a new sensor requires only adding a new entry here.
#
# Band index references:
#   Sentinel-2 L2A: B1(coastal), B2(Blue,490nm), B3(Green,560nm), B4(Red,665nm),
#                   B5(RedEdge1), B6(RedEdge2), B7(RedEdge3), B8(NIR,842nm),
#                   B8A(NarrowNIR), B9(WaterVapour), B11(SWIR1,1610nm), B12(SWIR2,2190nm)
#                   In GeoTIFF products, bands are typically ordered sequentially.
#   LISS-IV:        B2(Green,520-590nm), B3(Red,620-680nm), B4(NIR,770-860nm),
#                   B5(MIR,1550-1700nm)

SENSOR_REGISTRY: dict = {
    SensorID.SENTINEL2: SensorSpec(
        sensor_id=SensorID.SENTINEL2,
        display_name="Sentinel-2 (ESA Copernicus)",
        resolution_m=10.0,
        # RGB: in the processed GeoTIFF used by Version 1, bands are stored as
        # Band 1 = Red (B4), Band 2 = Green (B3), Band 3 = Blue (B2).
        rgb_band_indices=(1, 2, 3),
        # NDVI = (NIR - Red) / (NIR + Red)
        # In raw Sentinel-2 L2A multi-band TIFFs: B8=NIR at band 8, B4=Red at band 4.
        # Band indices below are for raw 12-band Sentinel-2 TIFFs.
        ndvi_bands=(8, 4),    # (NIR band index, Red band index)
        # NDBI = (SWIR - NIR) / (SWIR + NIR)
        # B11=SWIR1 at band 11, B8=NIR at band 8.
        ndbi_bands=(11, 8),   # (SWIR1 band index, NIR band index)
        # NDWI = (Green - NIR) / (Green + NIR)
        # B3=Green at band 3, B8=NIR at band 8.
        ndwi_bands=(3, 8),    # (Green band index, NIR band index)
        is_eo=True,
        is_sar=False,
        notes=[
            "Band indices are for raw Sentinel-2 L2A 12-band TIFF products.",
            "In Version 1, pre-computed single-band index rasters are loaded directly.",
            "RGB GeoTIFF in Version 1 stores Red=Band1, Green=Band2, Blue=Band3.",
            "Maximum image size: ~11000 x 11000 pixels at 10m resolution.",
        ],
    ),

    SensorID.SENTINEL1: SensorSpec(
        sensor_id=SensorID.SENTINEL1,
        display_name="Sentinel-1 C-band SAR (ESA Copernicus)",
        resolution_m=10.0,
        rgb_band_indices=None,
        ndvi_bands=None,
        ndbi_bands=None,
        ndwi_bands=None,
        vv_band_index=1,    # VV polarisation is stored in band 1.
        is_eo=False,
        is_sar=True,
        notes=[
            "VV polarisation is read from band 1.",
            "GRD (Ground Range Detected) products are used.",
            "Values are in dB (logarithmic scale) after preprocessing.",
        ],
    ),

    SensorID.LISS4: SensorSpec(
        sensor_id=SensorID.LISS4,
        display_name="ResourceSat-2 LISS-IV (ISRO/NRSC)",
        resolution_m=5.8,
        # LISS-IV band order in NRSC GeoTIFF products:
        # Band 1 = Green (520-590 nm)
        # Band 2 = Red (620-680 nm)
        # Band 3 = NIR (770-860 nm)
        # Band 4 = MIR/SWIR (1550-1700 nm)
        # NOTE: Band order confirmed from NRSC LISS-IV product specification.
        # Verify against actual downloaded data before using Mode 2.
        rgb_band_indices=(2, 1, 3),  # (Red=B2, Green=B1, Blue=None→NIR=B3 used as pseudo-blue)
        # NDVI = (NIR - Red) / (NIR + Red)  → B3=NIR, B2=Red
        ndvi_bands=(3, 2),
        # NDWI = (Green - NIR) / (Green + NIR)  → B1=Green, B3=NIR
        ndwi_bands=(1, 3),
        # NDBI approximation using MIR instead of SWIR:
        # NDBI = (MIR - NIR) / (MIR + NIR)  → B4=MIR, B3=NIR
        # Note: MIR (1550-1700nm) is a different spectral range from the
        # conventional SWIR (1610nm used with Sentinel-2 B11). Results may
        # differ numerically but the spectral intent is the same.
        ndbi_bands=(4, 3),
        is_eo=True,
        is_sar=False,
        notes=[
            "MISSING INFORMATION: Band order should be verified against actual",
            "LISS-IV GeoTIFF products downloaded from Bhoonidhi portal before",
            "using index_mode: compute with this sensor.",
            "Maximum image size: ~18000 x 17000 pixels at 5.8m resolution.",
            "RGB pseudo-colour: Red=B2, Green=B1, Blue=B3(NIR) — NIR used as proxy for Blue.",
            "NDBI uses MIR (B4) instead of conventional SWIR; numerical values will differ",
            "from Sentinel-2 NDBI but directional change signal is preserved.",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Registry access functions
# ---------------------------------------------------------------------------


def get_sensor_spec(sensor_id: str) -> SensorSpec:
    """Retrieve the sensor specification for a given sensor identifier.

    Args:
        sensor_id: Sensor identifier string. Must match a key in
            ``SENSOR_REGISTRY``. Case-sensitive. See :class:`SensorID` for
            canonical values.

    Returns:
        The :class:`SensorSpec` for the requested sensor.

    Raises:
        UnknownSensorError: If ``sensor_id`` is not registered.
    """
    spec = SENSOR_REGISTRY.get(sensor_id)
    if spec is None:
        raise UnknownSensorError(sensor_id)
    logger.debug("Sensor spec retrieved for '%s'.", sensor_id)
    return spec


def is_sensor_registered(sensor_id: str) -> bool:
    """Return True if a sensor identifier is present in the registry.

    Args:
        sensor_id: Sensor identifier string to check.

    Returns:
        True if the sensor is registered; False otherwise.
    """
    return sensor_id in SENSOR_REGISTRY


def list_registered_sensors() -> list:
    """Return a list of all registered sensor identifiers.

    Returns:
        List of sensor ID strings registered in ``SENSOR_REGISTRY``.
    """
    return list(SENSOR_REGISTRY.keys())


def get_eo_sensors() -> list:
    """Return sensor specifications for all registered electro-optical sensors.

    Returns:
        List of :class:`SensorSpec` instances where ``is_eo`` is True.
    """
    return [spec for spec in SENSOR_REGISTRY.values() if spec.is_eo]


def get_sar_sensors() -> list:
    """Return sensor specifications for all registered SAR sensors.

    Returns:
        List of :class:`SensorSpec` instances where ``is_sar`` is True.
    """
    return [spec for spec in SENSOR_REGISTRY.values() if spec.is_sar]
