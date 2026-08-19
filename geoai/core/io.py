"""
Raster I/O abstraction layer for the GeoAI Platform.

Provides thin, consistent wrappers around ``rasterio`` for reading and writing
raster files. All production modules use these wrappers rather than calling
``rasterio`` directly. This ensures consistent error handling, logging, and
metadata extraction across the platform.

Single responsibility: open, read, and write raster files.

Position in dependency hierarchy: core (depends on exceptions only).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from geoai.core.exceptions import RasterIOError, RasterNotFoundError, RasterWriteError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A raster profile is the dictionary of metadata parameters used by rasterio
# to describe and recreate a raster file.
RasterProfile = Dict


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_raster(
    path: Union[str, Path],
    band: int = 1,
    fill_nan: bool = True,
) -> Tuple[np.ndarray, RasterProfile]:
    """Read a single band from a raster file into a NumPy array.

    The returned array always has shape ``(height, width)`` and dtype
    ``float32``. NoData values are replaced with NaN unless ``fill_nan``
    is False, in which case they retain the raw nodata value stored in the
    file profile.

    Args:
        path: Path to the raster file. Accepts GeoTIFF (.tif/.tiff) and
            JPEG2000 (.jp2) formats.
        band: 1-indexed band number to read. Default is 1.
        fill_nan: If True (default), replace nodata-masked values with
            ``np.nan``. If False, return the raw values including nodata.

    Returns:
        A tuple of:
            - ``array``: NumPy float32 array of shape ``(height, width)``.
            - ``profile``: rasterio profile dictionary containing CRS,
              transform, dtype, nodata, width, height, and count.

    Raises:
        RasterNotFoundError: If the file does not exist at ``path``.
        RasterIOError: If the file cannot be opened or read.
    """
    path = Path(path)
    if not path.exists():
        raise RasterNotFoundError(str(path))

    logger.debug("Reading raster '%s' band %d.", path.name, band)

    try:
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            data = src.read(band).astype(np.float32)

            if fill_nan and src.nodata is not None:
                nodata_value = float(src.nodata)
                data[data == nodata_value] = np.nan
                logger.debug(
                    "Replaced nodata value %.4g with NaN in '%s'.",
                    nodata_value,
                    path.name,
                )
    except rasterio.errors.RasterioIOError as exc:
        raise RasterIOError(
            f"Failed to read raster '{path}': {exc}."
        ) from exc

    logger.debug(
        "Loaded '%s' — shape=%s dtype=%s min=%.4g max=%.4g nan_count=%d.",
        path.name,
        data.shape,
        data.dtype,
        float(np.nanmin(data)) if data.size > 0 else float("nan"),
        float(np.nanmax(data)) if data.size > 0 else float("nan"),
        int(np.isnan(data).sum()),
    )
    return data, profile


def read_raster_bands(
    path: Union[str, Path],
    bands: Optional[list] = None,
    fill_nan: bool = True,
) -> Tuple[np.ndarray, RasterProfile]:
    """Read multiple bands from a raster file into a 3D NumPy array.

    Args:
        path: Path to the raster file.
        bands: List of 1-indexed band numbers to read. If None, all bands
            are read.
        fill_nan: If True, replace nodata-masked values with ``np.nan``.

    Returns:
        A tuple of:
            - ``array``: NumPy float32 array of shape ``(bands, height, width)``.
            - ``profile``: rasterio profile dictionary.

    Raises:
        RasterNotFoundError: If the file does not exist at ``path``.
        RasterIOError: If the file cannot be opened or read.
    """
    path = Path(path)
    if not path.exists():
        raise RasterNotFoundError(str(path))

    logger.debug("Reading raster '%s' (multi-band).", path.name)

    try:
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            read_bands = bands if bands is not None else list(range(1, src.count + 1))
            data = src.read(read_bands).astype(np.float32)

            if fill_nan and src.nodata is not None:
                nodata_value = float(src.nodata)
                data[data == nodata_value] = np.nan
    except rasterio.errors.RasterioIOError as exc:
        raise RasterIOError(
            f"Failed to read multi-band raster '{path}': {exc}."
        ) from exc

    logger.debug(
        "Loaded '%s' — bands=%d shape=%s.", path.name, data.shape[0], data.shape
    )
    return data, profile


def read_raster_metadata(path: Union[str, Path]) -> RasterProfile:
    """Read raster metadata without loading pixel data.

    Useful for validation and compatibility checks before committing to
    loading large raster files into memory.

    Args:
        path: Path to the raster file.

    Returns:
        rasterio profile dictionary containing CRS, transform, dtype,
        nodata, width, height, and count.

    Raises:
        RasterNotFoundError: If the file does not exist at ``path``.
        RasterIOError: If the file cannot be opened.
    """
    path = Path(path)
    if not path.exists():
        raise RasterNotFoundError(str(path))

    try:
        with rasterio.open(path) as src:
            profile = src.profile.copy()
            # Supplement the standard profile with derived metadata.
            profile["bounds"] = src.bounds
            profile["res"] = src.res
    except rasterio.errors.RasterioIOError as exc:
        raise RasterIOError(
            f"Failed to read metadata from '{path}': {exc}."
        ) from exc

    logger.debug(
        "Metadata for '%s': width=%d height=%d crs=%s res=%s.",
        path.name,
        profile["width"],
        profile["height"],
        profile.get("crs"),
        profile.get("res"),
    )
    return profile


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_raster(
    path: Union[str, Path],
    array: np.ndarray,
    profile: RasterProfile,
    compress: str = "lzw",
    nodata: Optional[float] = None,
) -> Path:
    """Write a 2D NumPy array to a GeoTIFF file.

    The output file uses the CRS and affine transform from ``profile``.
    The parent directory is created if it does not exist.

    Args:
        path: Destination file path. Extension should be ``.tif`` or ``.tiff``.
        array: 2D NumPy array of shape ``(height, width)`` to write. Will be
            cast to the dtype specified in ``profile``, or to ``float32`` if
            the profile does not specify a dtype.
        profile: rasterio profile dictionary providing at minimum ``crs``,
            ``transform``, ``width``, and ``height``.
        compress: Compression algorithm. Default ``'lzw'`` is recommended
            for binary and integer rasters.
        nodata: Value to use as the nodata sentinel. If None, no nodata value
            is written unless the profile already contains one.

    Returns:
        The resolved absolute :class:`pathlib.Path` of the written file.

    Raises:
        RasterWriteError: If the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_profile = profile.copy()
    write_profile.update(
        driver="GTiff",
        count=1,
        compress=compress,
    )
    if nodata is not None:
        write_profile["nodata"] = nodata

    out_dtype = write_profile.get("dtype", "float32")
    out_array = array.astype(out_dtype)

    logger.info(
        "Writing raster to '%s' — shape=%s dtype=%s compress=%s.",
        path,
        out_array.shape,
        out_dtype,
        compress,
    )

    try:
        with rasterio.open(path, "w", **write_profile) as dst:
            dst.write(out_array, 1)
    except Exception as exc:
        raise RasterWriteError(str(path), str(exc)) from exc

    logger.info("Raster written successfully: '%s'.", path)
    return path.resolve()


def write_binary_mask(
    path: Union[str, Path],
    mask: np.ndarray,
    profile: RasterProfile,
    compress: str = "lzw",
) -> Path:
    """Write a binary change mask as a GeoTIFF with uint8 dtype.

    Pixel value 1 represents change; pixel value 0 represents no change.
    This is the PS10 challenge output format.

    Args:
        path: Destination file path.
        mask: 2D NumPy array of shape ``(height, width)`` with values 0 and 1.
            Any non-zero value is treated as change.
        profile: rasterio profile dictionary providing CRS and transform.
        compress: Compression algorithm. Default ``'lzw'``.

    Returns:
        The resolved absolute :class:`pathlib.Path` of the written file.

    Raises:
        RasterWriteError: If the file cannot be written.
    """
    binary = (mask > 0).astype(np.uint8)

    write_profile = profile.copy()
    write_profile.update(dtype="uint8", nodata=None)

    logger.info(
        "Writing binary change mask to '%s' — change_pixels=%d total_pixels=%d.",
        path,
        int(binary.sum()),
        binary.size,
    )
    return write_raster(
        path=path,
        array=binary,
        profile=write_profile,
        compress=compress,
        nodata=None,
    )


def write_multiband_raster(
    path: Union[str, Path],
    array: np.ndarray,
    profile: RasterProfile,
    compress: str = "lzw",
) -> Path:
    """Write a 3D NumPy array as a multi-band GeoTIFF.

    Args:
        path: Destination file path.
        array: 3D NumPy array of shape ``(bands, height, width)`` or
            ``(height, width, bands)``. If the last axis is the band axis
            (shape ``(H, W, C)``), it is transposed automatically.
        profile: rasterio profile dictionary providing CRS and transform.
        compress: Compression algorithm.

    Returns:
        The resolved absolute :class:`pathlib.Path` of the written file.

    Raises:
        RasterWriteError: If the file cannot be written.
    """
    # Accept (H, W, C) and convert to rasterio's expected (C, H, W).
    if array.ndim == 3 and array.shape[2] < array.shape[0]:
        array = np.transpose(array, (2, 0, 1))

    if array.ndim != 3:
        raise RasterWriteError(
            str(path),
            f"Expected a 3D array for multi-band write, got shape {array.shape}.",
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    write_profile = profile.copy()
    write_profile.update(
        driver="GTiff",
        count=array.shape[0],
        dtype=str(array.dtype),
        compress=compress,
    )

    logger.info(
        "Writing %d-band raster to '%s' — shape=%s.",
        array.shape[0],
        path,
        array.shape,
    )

    try:
        with rasterio.open(path, "w", **write_profile) as dst:
            dst.write(array)
    except Exception as exc:
        raise RasterWriteError(str(path), str(exc)) from exc

    logger.info("Multi-band raster written: '%s'.", path)
    return path.resolve()
