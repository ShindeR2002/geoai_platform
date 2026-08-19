"""
Raster compatibility validation for the GeoAI Platform.

Validates that rasters intended for joint processing share the same coordinate
reference system, spatial dimensions, and affine transform. These checks are
mandatory before any feature cube assembly and must be called by every pipeline
stage that loads multiple raster files.

Single responsibility: validate raster spatial compatibility.

Position in dependency hierarchy: core (depends on exceptions and io).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import rasterio
from rasterio.transform import Affine

from geoai.core.exceptions import (
    CRSMismatchError,
    RasterNotFoundError,
    RasterValidationError,
    SpatialDimensionMismatchError,
    TransformMismatchError,
)
from geoai.core.io import read_raster_metadata, RasterProfile

logger = logging.getLogger(__name__)

# Tolerance for comparing affine transform coefficients. Two transforms are
# considered identical if all six coefficients agree within this tolerance.
TRANSFORM_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Primary validation entry point
# ---------------------------------------------------------------------------


def validate_raster_compatibility(
    raster_paths: Dict[str, Union[str, Path]],
    reference_key: Optional[str] = None,
) -> Dict[str, RasterProfile]:
    """Validate that a collection of raster files are spatially compatible.

    Checks that all rasters share the same:
    - Coordinate reference system (CRS),
    - Spatial dimensions (height × width),
    - Affine transform (origin and pixel size).

    This function must be called before any feature cube assembly to ensure
    that pixel positions in different rasters correspond to the same geographic
    location.

    Args:
        raster_paths: Dictionary mapping descriptive names (e.g. 'RGB_T1',
            'NDVI_T2') to file paths. Names are used in error messages.
        reference_key: The key in ``raster_paths`` to use as the reference
            raster against which all others are compared. If None, the first
            key in dictionary insertion order is used as the reference.

    Returns:
        Dictionary mapping the same descriptive names to their rasterio
        profile dictionaries. Profiles are returned so callers can reuse
        them for output file creation without reopening files.

    Raises:
        RasterNotFoundError: If any raster file does not exist.
        CRSMismatchError: If any raster's CRS differs from the reference.
        SpatialDimensionMismatchError: If any raster's dimensions differ from
            the reference.
        TransformMismatchError: If any raster's affine transform differs from
            the reference beyond ``TRANSFORM_TOLERANCE``.
        RasterValidationError: For any other compatibility failure.
    """
    if not raster_paths:
        raise RasterValidationError(
            "raster_paths must contain at least one entry."
        )

    logger.info(
        "Validating spatial compatibility of %d rasters.", len(raster_paths)
    )

    # Load all metadata first so we fail fast on missing files.
    profiles: Dict[str, RasterProfile] = {}
    for name, path in raster_paths.items():
        logger.debug("Loading metadata for '%s' from '%s'.", name, path)
        profiles[name] = read_raster_metadata(path)

    keys = list(profiles.keys())
    ref_key = reference_key if reference_key is not None else keys[0]
    if ref_key not in profiles:
        raise RasterValidationError(
            f"Reference key '{ref_key}' not found in raster_paths. "
            f"Available keys: {keys}."
        )

    ref_profile = profiles[ref_key]
    ref_crs = ref_profile.get("crs")
    ref_width = ref_profile["width"]
    ref_height = ref_profile["height"]
    ref_transform = ref_profile["transform"]

    logger.info(
        "Reference raster '%s': width=%d height=%d crs=%s.",
        ref_key,
        ref_width,
        ref_height,
        ref_crs,
    )

    for name, profile in profiles.items():
        if name == ref_key:
            continue

        # CRS check
        crs = profile.get("crs")
        if ref_crs is not None and crs is not None:
            if str(crs) != str(ref_crs):
                raise CRSMismatchError(
                    name_a=ref_key,
                    crs_a=str(ref_crs),
                    name_b=name,
                    crs_b=str(crs),
                )
        elif ref_crs != crs:
            # One is None and the other is not — treat as mismatch.
            raise CRSMismatchError(
                name_a=ref_key,
                crs_a=str(ref_crs),
                name_b=name,
                crs_b=str(crs),
            )

        # Spatial dimensions check
        width = profile["width"]
        height = profile["height"]
        if (width, height) != (ref_width, ref_height):
            raise SpatialDimensionMismatchError(
                name_a=ref_key,
                shape_a=(ref_height, ref_width),
                name_b=name,
                shape_b=(height, width),
            )

        # Affine transform check
        transform = profile["transform"]
        if not _transforms_equal(ref_transform, transform):
            raise TransformMismatchError(name_a=ref_key, name_b=name)

        logger.debug("'%s' passed compatibility check against '%s'.", name, ref_key)

    logger.info(
        "All %d rasters passed spatial compatibility validation.",
        len(raster_paths),
    )
    return profiles


# ---------------------------------------------------------------------------
# Band count validation
# ---------------------------------------------------------------------------


def validate_band_count(
    path: Union[str, Path],
    expected_count: int,
    name: str = "raster",
) -> None:
    """Assert that a raster file contains the expected number of bands.

    Args:
        path: Path to the raster file.
        expected_count: The number of bands required.
        name: Descriptive name for the raster, used in error messages.

    Raises:
        RasterNotFoundError: If the file does not exist.
        RasterValidationError: If the band count does not match.
    """
    profile = read_raster_metadata(path)
    actual_count = profile["count"]
    if actual_count != expected_count:
        raise RasterValidationError(
            f"Raster '{name}' at '{path}' has {actual_count} band(s) but "
            f"{expected_count} band(s) are required."
        )
    logger.debug("'%s' band count check passed: %d band(s).", name, actual_count)


# ---------------------------------------------------------------------------
# File existence validation
# ---------------------------------------------------------------------------


def validate_raster_exists(
    path: Union[str, Path],
    name: str = "raster",
) -> None:
    """Assert that a raster file exists at the specified path.

    Args:
        path: Path to check.
        name: Descriptive name for the raster, used in error messages.

    Raises:
        RasterNotFoundError: If the file does not exist.
    """
    if not Path(path).exists():
        raise RasterNotFoundError(str(path))
    logger.debug("File existence confirmed for '%s': '%s'.", name, path)


def validate_all_rasters_exist(
    raster_paths: Dict[str, Union[str, Path]],
) -> None:
    """Assert that all rasters in a dictionary exist on the file system.

    Collects all missing paths before raising so the caller receives a complete
    list of missing files rather than discovering them one at a time.

    Args:
        raster_paths: Dictionary mapping descriptive names to file paths.

    Raises:
        RasterNotFoundError: If any raster file is missing. The error message
            lists all missing files.
    """
    missing = {
        name: str(path)
        for name, path in raster_paths.items()
        if not Path(path).exists()
    }
    if missing:
        missing_list = "\n  ".join(
            f"'{name}': {path}" for name, path in missing.items()
        )
        raise RasterNotFoundError(
            f"The following required raster files were not found:\n  {missing_list}"
        )
    logger.debug(
        "All %d raster files confirmed to exist.", len(raster_paths)
    )


# ---------------------------------------------------------------------------
# Profile summary utility
# ---------------------------------------------------------------------------


def summarise_raster_profile(name: str, profile: RasterProfile) -> str:
    """Format a human-readable summary of a raster profile.

    Args:
        name: Descriptive name for the raster.
        profile: rasterio profile dictionary.

    Returns:
        A single-line string suitable for logging.
    """
    return (
        f"'{name}': width={profile.get('width')} height={profile.get('height')} "
        f"bands={profile.get('count')} dtype={profile.get('dtype')} "
        f"crs={profile.get('crs')} res={profile.get('res')}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _transforms_equal(t1: Affine, t2: Affine, tol: float = TRANSFORM_TOLERANCE) -> bool:
    """Compare two affine transforms for equality within a tolerance.

    Args:
        t1: First affine transform.
        t2: Second affine transform.
        tol: Maximum allowed difference between corresponding coefficients.

    Returns:
        True if all six transform coefficients agree within ``tol``.
    """
    coeffs_a = (t1.a, t1.b, t1.c, t1.d, t1.e, t1.f)
    coeffs_b = (t2.a, t2.b, t2.c, t2.d, t2.e, t2.f)
    return all(abs(a - b) <= tol for a, b in zip(coeffs_a, coeffs_b))
