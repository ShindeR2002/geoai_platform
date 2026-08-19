"""
Layer and AOI management for the GeoAI Platform dashboard backend.

Provides the data layer and registry that the dashboard frontend (or API)
uses to discover available raster layers, vector layers, and AOI metadata
for a given pipeline run.

These modules do not implement a frontend. They produce the data contracts
that any frontend (Leaflet, Mapbox, Kepler.gl) can consume.

Single responsibility: maintain registries of layers and AOI metadata
for dashboard consumption.

Position in dependency hierarchy: dashboard (depends on core/config).
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer registry
# ---------------------------------------------------------------------------


@dataclass
class LayerRecord:
    """Metadata record for a single raster or vector layer.

    Attributes:
        layer_id: Unique layer identifier.
        layer_type: One of 'raster', 'vector', 'geojson'.
        name: Human-readable layer name.
        file_path: Path to the layer file on disk.
        run_id: Run that produced this layer.
        aoi_name: AOI this layer belongs to.
        description: Optional description for the frontend.
        metadata: Optional additional metadata dictionary.
    """

    layer_id: str
    layer_type: str
    name: str
    file_path: str
    run_id: str
    aoi_name: str
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayerManager:
    """Maintains a registry of raster and vector layers for dashboard consumption.

    Layers are registered by pipeline stages and can be queried by run ID
    or AOI name. The registry is persisted to a JSON file.

    Args:
        registry_path: Path to the layer registry JSON file.
    """

    def __init__(self, registry_path: str = "outputs/layer_registry.json") -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._layers: Dict[str, LayerRecord] = {}
        self._load_registry()
        logger.info(
            "LayerManager initialised — %d layers in registry.",
            len(self._layers),
        )

    def register_layer(self, record: LayerRecord) -> None:
        """Add a layer to the registry.

        Args:
            record: LayerRecord describing the layer.
        """
        self._layers[record.layer_id] = record
        self._save_registry()
        logger.debug("Layer registered: '%s' (%s).", record.layer_id, record.name)

    def get_layers_for_run(self, run_id: str) -> List[LayerRecord]:
        """Return all layers produced by a specific run.

        Args:
            run_id: Run identifier to filter by.

        Returns:
            List of LayerRecord instances for the specified run.
        """
        return [r for r in self._layers.values() if r.run_id == run_id]

    def get_layers_for_aoi(self, aoi_name: str) -> List[LayerRecord]:
        """Return all layers for a specific AOI.

        Args:
            aoi_name: AOI name to filter by.

        Returns:
            List of LayerRecord instances for the specified AOI.
        """
        return [r for r in self._layers.values() if r.aoi_name == aoi_name]

    def to_frontend_payload(self, run_id: Optional[str] = None) -> List[Dict]:
        """Serialise layers to a JSON-ready list for frontend consumption.

        Args:
            run_id: Optional run ID to filter. If None, return all layers.

        Returns:
            List of dictionaries describing each layer.
        """
        layers = (
            self.get_layers_for_run(run_id) if run_id else list(self._layers.values())
        )
        return [asdict(layer) for layer in layers]

    def _load_registry(self) -> None:
        """Load the persisted registry from disk."""
        if not self.registry_path.exists():
            return
        try:
            with open(self.registry_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._layers = {k: LayerRecord(**v) for k, v in raw.items()}
        except Exception as exc:
            logger.warning(
                "Could not load layer registry from '%s': %s. "
                "Starting with empty registry.",
                self.registry_path, exc,
            )

    def _save_registry(self) -> None:
        """Persist the current registry to disk."""
        with open(self.registry_path, "w", encoding="utf-8") as fh:
            json.dump(
                {k: asdict(v) for k, v in self._layers.items()},
                fh, indent=2,
            )


# ---------------------------------------------------------------------------
# AOI manager
# ---------------------------------------------------------------------------


@dataclass
class AOIRecord:
    """Metadata record for a registered AOI.

    Attributes:
        aoi_name: Canonical AOI name.
        display_name: Human-readable name for UI display.
        reference_lat: Reference latitude (from PS10 dataset table or estimated).
        reference_lon: Reference longitude.
        t1_year: Epoch T1 year.
        t2_year: Epoch T2 year.
        sensor_eo: Electro-optical sensor identifier.
        sensor_sar: SAR sensor identifier.
        resolution_m: Ground sampling distance in metres.
        crs: Coordinate reference system string.
        run_count: Number of pipeline runs completed for this AOI.
        last_run_id: Most recent run identifier.
    """

    aoi_name: str
    display_name: str
    reference_lat: Optional[float]
    reference_lon: Optional[float]
    t1_year: int
    t2_year: int
    sensor_eo: str = "sentinel2"
    sensor_sar: str = "sentinel1"
    resolution_m: float = 10.0
    crs: str = ""
    run_count: int = 0
    last_run_id: str = ""


class AOIManager:
    """Manages AOI metadata and boundary information for the dashboard.

    Provides discovery, registration, and frontend serialisation for all
    AOIs that have been processed by the platform.

    Args:
        registry_path: Path to the AOI registry JSON file.
    """

    def __init__(self, registry_path: str = "outputs/aoi_registry.json") -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._aois: Dict[str, AOIRecord] = {}
        self._load()
        logger.info(
            "AOIManager initialised — %d AOIs registered.", len(self._aois)
        )

    def register_aoi_from_config(self, aoi_config) -> None:
        """Register an AOI from a platform AOIConfig instance.

        Args:
            aoi_config: :class:`~geoai.core.config.AOIConfig` instance.
        """
        record = AOIRecord(
            aoi_name=aoi_config.name,
            display_name=aoi_config.name.replace("_", " "),
            reference_lat=aoi_config.reference_lat,
            reference_lon=aoi_config.reference_lon,
            t1_year=aoi_config.t1_year,
            t2_year=aoi_config.t2_year,
            sensor_eo=aoi_config.sensor_eo,
            sensor_sar=aoi_config.sensor_sar or "",
            resolution_m=aoi_config.resolution_m,
            crs=aoi_config.crs or "",
        )
        self._aois[aoi_config.name] = record
        self._save()
        logger.debug("AOI '%s' registered.", aoi_config.name)

    def update_run_info(self, aoi_name: str, run_id: str) -> None:
        """Increment run count and update last run ID for an AOI.

        Args:
            aoi_name: AOI to update.
            run_id: Most recent run identifier.
        """
        if aoi_name in self._aois:
            self._aois[aoi_name].run_count += 1
            self._aois[aoi_name].last_run_id = run_id
            self._save()

    def get_all_aois(self) -> List[Dict]:
        """Return all registered AOIs as a list of dictionaries.

        Returns:
            List of AOI record dictionaries for API/frontend consumption.
        """
        return [asdict(r) for r in self._aois.values()]

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            with open(self.registry_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._aois = {k: AOIRecord(**v) for k, v in raw.items()}
        except Exception as exc:
            logger.warning("Could not load AOI registry: %s.", exc)

    def _save(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as fh:
            json.dump(
                {k: asdict(v) for k, v in self._aois.items()},
                fh, indent=2,
            )
