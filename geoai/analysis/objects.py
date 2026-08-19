"""
Spatial object extraction for the GeoAI Platform.

Extracts discrete change objects from the binary change mask using connected
component labelling and filters objects by minimum area.

Implements the exact Version 1 NB06 / NB09 object extraction workflow:

    labels = skimage.measure.label(binary_change, connectivity=2)
    regions = skimage.measure.regionprops(labels)
    significant = [r for r in regions if r.area >= MIN_OBJECT_SIZE]

Parameters (frozen by Version 1):
  - connectivity=2 (8-connectivity — diagonal neighbours included)
  - MIN_OBJECT_SIZE=50 pixels (configurable default)

Single responsibility: label connected components and filter by area.

Position in dependency hierarchy: analysis (depends on core, utils, postprocessing).
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties

from geoai.core.exceptions import GeoAIPlatformError
from geoai.utils.constants import DEFAULT_MIN_OBJECT_SIZE_PX, OBJECT_CONNECTIVITY

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connected component labelling
# ---------------------------------------------------------------------------


def label_connected_components(binary_mask: np.ndarray) -> np.ndarray:
    """Label connected components in a binary change mask.

    Uses 8-connectivity (connectivity=2) as specified in Version 1 NB06 Cell 3
    and NB09 Cell 17. Diagonal neighbours are considered connected.

    Args:
        binary_mask: uint8 binary array of shape (H, W) with values 0 and 1.

    Returns:
        Integer label array of shape (H, W) where each connected component
        has a unique positive integer label. Background pixels (no change)
        have label 0.
    """
    logger.info(
        "Labelling connected components — input shape=%s change_px=%d.",
        binary_mask.shape,
        int(binary_mask.sum()),
    )

    # connectivity=2 → 8-connectivity (includes diagonals).
    # This is the canonical setting from V1 NB06 Cell 3 and NB09 Cell 17.
    labels = label(binary_mask, connectivity=OBJECT_CONNECTIVITY)
    n_components = int(labels.max())

    logger.info(
        "Connected component labelling complete — %d components found.",
        n_components,
    )
    return labels


# ---------------------------------------------------------------------------
# Region property extraction
# ---------------------------------------------------------------------------


def extract_region_properties(label_array: np.ndarray) -> List[RegionProperties]:
    """Extract region properties for all labelled components.

    Args:
        label_array: Integer label array from :func:`label_connected_components`.

    Returns:
        List of :class:`skimage.measure._regionprops.RegionProperties` objects,
        one per connected component (excluding background label 0).
    """
    regions = regionprops(label_array)
    logger.debug(
        "Region properties extracted for %d components.", len(regions)
    )
    return regions


# ---------------------------------------------------------------------------
# Area filtering (canonical Version 1 operation)
# ---------------------------------------------------------------------------


def filter_objects_by_area(
    regions: List[RegionProperties],
    min_area_px: int = DEFAULT_MIN_OBJECT_SIZE_PX,
) -> List[RegionProperties]:
    """Filter region objects to retain only those meeting the minimum area threshold.

    Implements the Version 1 NB06 Cell 8 / NB09 Cell 18 filtering:

        significant = [r for r in regions if r.area >= MIN_OBJECT_SIZE]

    Objects smaller than ``min_area_px`` are discarded as noise.

    Args:
        regions: List of RegionProperties from :func:`extract_region_properties`.
        min_area_px: Minimum object area in pixels. Objects with area strictly
            less than this value are excluded. Default 50 (V1 NB06/NB09).

    Returns:
        Filtered list of RegionProperties where every object has
        ``area >= min_area_px``.
    """
    n_before = len(regions)
    significant = [r for r in regions if r.area >= min_area_px]
    n_after = len(significant)

    if n_before > 0:
        logger.info(
            "Object area filter (min=%d px): %d → %d objects retained "
            "(%.1f%% passed).",
            min_area_px, n_before, n_after,
            100.0 * n_after / n_before,
        )
    else:
        logger.warning(
            "Object area filter: no objects to filter (input list is empty)."
        )

    return significant


# ---------------------------------------------------------------------------
# Significant change mask reconstruction
# ---------------------------------------------------------------------------


def build_significant_mask(
    label_array: np.ndarray,
    significant_regions: List[RegionProperties],
) -> np.ndarray:
    """Build a binary mask containing only the significant (filtered) objects.

    Pixels belonging to significant objects have value 1; all other pixels
    (including small objects that were filtered out) have value 0.

    Args:
        label_array: Integer label array of shape (H, W).
        significant_regions: Filtered list from :func:`filter_objects_by_area`.

    Returns:
        uint8 binary array of shape (H, W) containing only significant objects.
    """
    significant_labels = {r.label for r in significant_regions}
    significant_mask = np.isin(label_array, list(significant_labels)).astype(np.uint8)

    logger.info(
        "Significant mask built — %d objects, %d change pixels.",
        len(significant_regions),
        int(significant_mask.sum()),
    )
    return significant_mask


# ---------------------------------------------------------------------------
# Primary entry point (used by pipeline)
# ---------------------------------------------------------------------------


def extract_objects(
    binary_mask: np.ndarray,
    min_area_px: int = DEFAULT_MIN_OBJECT_SIZE_PX,
) -> Tuple[List[RegionProperties], np.ndarray, np.ndarray]:
    """Extract significant change objects from a binary change mask.

    This is the single entry point for object extraction used by all pipeline
    stages. It chains labelling, property extraction, area filtering, and
    significant mask construction.

    Implements the complete Version 1 NB06 / NB09 object extraction workflow.

    Args:
        binary_mask: uint8 binary array of shape (H, W).
        min_area_px: Minimum object area in pixels. Default 50 (V1 NB06/NB09).

    Returns:
        Tuple of three elements:
            - ``significant_regions``: List of RegionProperties for objects
              with area >= min_area_px.
            - ``label_array``: Integer label array of shape (H, W) where
              each component has a unique label (including small objects).
            - ``significant_mask``: uint8 binary array of shape (H, W)
              containing only significant objects.
    """
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)

    label_array = label_connected_components(binary_mask)
    all_regions = extract_region_properties(label_array)
    significant_regions = filter_objects_by_area(all_regions, min_area_px)
    significant_mask = build_significant_mask(label_array, significant_regions)

    logger.info(
        "Object extraction complete — %d significant objects "
        "(>= %d px), total area = %d px.",
        len(significant_regions),
        min_area_px,
        int(significant_mask.sum()),
    )
    return significant_regions, label_array, significant_mask
