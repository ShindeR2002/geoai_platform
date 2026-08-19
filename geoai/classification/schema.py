"""
ChangeObject schema for the GeoAI Platform.

Defines the canonical data structure for a detected change object after
Stage 2 semantic classification. All Stage 2 modules produce and consume
ChangeObject instances, and all export modules serialise them to their
respective output formats.

This schema is the single source of truth for what attributes a classified
change object carries through the platform.

Single responsibility: define the ChangeObject dataclass and its serialisation.

Position in dependency hierarchy: classification (depends on utils/constants).
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from geoai.utils.constants import ChangeClass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ChangeObject dataclass
# ---------------------------------------------------------------------------


@dataclass
class ChangeObject:
    """Canonical representation of a detected and classified change object.

    Populated progressively as the object passes through the Stage 2 pipeline:
    - Core spatial attributes set by :mod:`geoai.analysis.statistics`.
    - Semantic class and confidence set by :mod:`geoai.classification.classifier`.
    - Spectral attributes set by :mod:`geoai.classification.object_features`.
    - SHAP values set by :mod:`geoai.classification.explainability` (optional).

    Attributes:
        object_id: Unique integer label assigned by connected component labelling.
        area_px: Object area in pixels.
        area_m2: Object area in square metres.
        centroid_row: Pixel row of the object centroid.
        centroid_col: Pixel column of the object centroid.
        centroid_lat: Geographic latitude of the centroid (None if unavailable).
        centroid_lon: Geographic longitude of the centroid (None if unavailable).
        bbox_min_row: Bounding box top row (pixel).
        bbox_min_col: Bounding box left column (pixel).
        bbox_max_row: Bounding box bottom row (pixel).
        bbox_max_col: Bounding box right column (pixel).
        perimeter: Object perimeter in pixels.
        compactness: Shape compactness = 4π×area / perimeter². Near 1 = circular.
        semantic_class: Assigned PS10 Stage 2 semantic class string.
        confidence: Confidence score in [0.0, 1.0].
        mean_ndvi_change: Mean Delta_NDVI within the object boundary.
        mean_ndbi_change: Mean Delta_NDBI within the object boundary.
        mean_ndwi_change: Mean Delta_NDWI within the object boundary.
        mean_sar_change: Mean Delta_SAR within the object boundary.
        mean_red_t2: Mean Red band (T2) within the object boundary.
        mean_ndvi_t2: Mean NDVI (T2) within the object boundary.
        shap_values: Per-feature SHAP attribution dict (optional).
        classifier_method: Name of the classifier that assigned semantic_class.
    """

    # --- Core spatial attributes (from Stage 1 object analysis) ---
    object_id: int = 0
    area_px: int = 0
    area_m2: float = 0.0
    centroid_row: float = 0.0
    centroid_col: float = 0.0
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    bbox_min_row: int = 0
    bbox_min_col: int = 0
    bbox_max_row: int = 0
    bbox_max_col: int = 0
    perimeter: float = 0.0
    compactness: float = 0.0

    # --- Stage 2 semantic classification ---
    semantic_class: str = ChangeClass.UNKNOWN
    confidence: float = 0.0

    # --- Spectral change attributes (from object_features.py) ---
    mean_ndvi_change: Optional[float] = None
    mean_ndbi_change: Optional[float] = None
    mean_ndwi_change: Optional[float] = None
    mean_sar_change: Optional[float] = None
    mean_red_t2: Optional[float] = None
    mean_ndvi_t2: Optional[float] = None

    # --- Explainability (from explainability.py, optional) ---
    shap_values: Optional[Dict[str, float]] = field(default=None, repr=False)

    # --- Provenance ---
    classifier_method: str = "unclassified"

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the ChangeObject to a plain dictionary.

        Returns:
            Dictionary with all attribute names and values. SHAP values are
            excluded if None to keep export payloads compact.
        """
        d = asdict(self)
        if d.get("shap_values") is None:
            del d["shap_values"]
        return d

    def to_export_dict(self) -> Dict[str, Any]:
        """Serialise the ChangeObject for shapefile / GeoJSON attribute export.

        Excludes non-serialisable or redundant fields (SHAP values, pixel
        coordinates when geographic coordinates are available). Rounds floats
        to 6 decimal places for compact output.

        Returns:
            Dictionary suitable for use as a GIS feature attribute table row.
        """
        return {
            "object_id": self.object_id,
            "area_px": self.area_px,
            "area_m2": round(self.area_m2, 2),
            "centroid_lat": round(self.centroid_lat, 6) if self.centroid_lat else None,
            "centroid_lon": round(self.centroid_lon, 6) if self.centroid_lon else None,
            "class": self.semantic_class,
            "confidence": round(self.confidence, 4),
            "ndvi_chg": round(self.mean_ndvi_change, 4) if self.mean_ndvi_change is not None else None,
            "ndbi_chg": round(self.mean_ndbi_change, 4) if self.mean_ndbi_change is not None else None,
            "ndwi_chg": round(self.mean_ndwi_change, 4) if self.mean_ndwi_change is not None else None,
            "sar_chg": round(self.mean_sar_change, 4) if self.mean_sar_change is not None else None,
            "compact": round(self.compactness, 4),
            "perimeter": round(self.perimeter, 2),
            "method": self.classifier_method,
        }


# ---------------------------------------------------------------------------
# Factory from statistics records
# ---------------------------------------------------------------------------


def change_object_from_record(record: Dict[str, Any]) -> ChangeObject:
    """Create a ChangeObject from a statistics module object record dictionary.

    Converts the output of :func:`~geoai.analysis.statistics.build_object_records`
    into a ChangeObject. Semantic class, confidence, and spectral attributes
    are left at their defaults and populated by the Stage 2 classifier.

    Args:
        record: Dictionary from ``build_object_records``.

    Returns:
        A ChangeObject populated with spatial attributes.
    """
    return ChangeObject(
        object_id=record["object_id"],
        area_px=record["area_px"],
        area_m2=record["area_m2"],
        centroid_row=record["centroid_row"],
        centroid_col=record["centroid_col"],
        centroid_lat=record.get("centroid_lat"),
        centroid_lon=record.get("centroid_lon"),
        bbox_min_row=record["bbox_min_row"],
        bbox_min_col=record["bbox_min_col"],
        bbox_max_row=record["bbox_max_row"],
        bbox_max_col=record["bbox_max_col"],
        perimeter=record.get("perimeter", 0.0),
        compactness=record.get("compactness", 0.0),
    )


def change_objects_from_records(records: List[Dict[str, Any]]) -> List[ChangeObject]:
    """Convert a list of object records to ChangeObject instances.

    Args:
        records: List of dictionaries from ``build_object_records``.

    Returns:
        List of ChangeObject instances with spatial attributes populated.
    """
    objects = [change_object_from_record(r) for r in records]
    logger.info("Created %d ChangeObject instances from records.", len(objects))
    return objects
