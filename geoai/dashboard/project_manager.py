"""
Project manager for the GeoAI Platform dashboard backend.

Manages projects as named collections of AOI runs. Each project persists its
state to a JSON manifest file, enabling run history tracking, status queries,
and project-level reporting across pipeline sessions.

Single responsibility: create, update, and query project manifests.

Position in dependency hierarchy: dashboard (depends on core/exceptions, utils).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from geoai.core.exceptions import GeoAIPlatformError

logger = logging.getLogger(__name__)


class ProjectManager:
    """Manages GeoAI Platform project state via JSON manifest files.

    Each project is represented by a JSON file at
    ``{manifest_dir}/{project_id}.json`` containing the project metadata,
    AOI list, run history, and aggregated statistics.

    Args:
        manifest_dir: Directory where project manifest files are stored.
    """

    def __init__(self, manifest_dir: str = "outputs/projects") -> None:
        self.manifest_dir = Path(manifest_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "ProjectManager initialised — manifest_dir='%s'.", self.manifest_dir
        )

    # -----------------------------------------------------------------------
    # Project lifecycle
    # -----------------------------------------------------------------------

    def create_project(
        self,
        project_id: str,
        aoi_names: Optional[List[str]] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """Create a new project manifest.

        If a manifest already exists for ``project_id``, it is loaded and
        returned without overwriting existing data.

        Args:
            project_id: Unique project identifier.
            aoi_names: List of AOI names this project will process.
            description: Optional human-readable description.

        Returns:
            The project manifest dictionary.
        """
        manifest_path = self._manifest_path(project_id)

        if manifest_path.exists():
            logger.info(
                "Project '%s' already exists — loading existing manifest.",
                project_id,
            )
            return self.load_project(project_id)

        manifest = {
            "project_id": project_id,
            "description": description,
            "created_utc": datetime.utcnow().isoformat(),
            "updated_utc": datetime.utcnow().isoformat(),
            "aoi_names": aoi_names or [],
            "runs": [],
            "failed_runs": [],
            "summary": {},
        }
        self._save_manifest(project_id, manifest)
        logger.info("Project '%s' created.", project_id)
        return manifest

    def load_project(self, project_id: str) -> Dict[str, Any]:
        """Load a project manifest from disk.

        Args:
            project_id: Project identifier.

        Returns:
            The project manifest dictionary.

        Raises:
            GeoAIPlatformError: If no manifest exists for this project.
        """
        manifest_path = self._manifest_path(project_id)
        if not manifest_path.exists():
            raise GeoAIPlatformError(
                f"No manifest found for project '{project_id}'. "
                f"Expected at '{manifest_path}'."
            )
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        logger.debug("Project '%s' manifest loaded.", project_id)
        return manifest

    def list_projects(self) -> List[str]:
        """Return a list of all project IDs with existing manifests.

        Returns:
            Sorted list of project ID strings.
        """
        return sorted(
            p.stem for p in self.manifest_dir.glob("*.json")
        )

    # -----------------------------------------------------------------------
    # Run tracking
    # -----------------------------------------------------------------------

    def record_run(
        self,
        project_id: str,
        aoi_name: str,
        run_id: str,
        stage1_summary: Dict[str, Any],
        object_count: int,
    ) -> None:
        """Record a completed pipeline run in the project manifest.

        Args:
            project_id: Project identifier.
            aoi_name: AOI that was processed.
            run_id: Run identifier.
            stage1_summary: Prediction summary from Stage 1.
            object_count: Number of significant objects detected.
        """
        manifest = self.load_project(project_id)
        run_record = {
            "run_id": run_id,
            "aoi_name": aoi_name,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "status": "success",
            "change_area_ha": stage1_summary.get("change_area_ha", 0.0),
            "change_pixels": stage1_summary.get("change_pixels", 0),
            "object_count": object_count,
        }
        manifest["runs"].append(run_record)
        manifest["updated_utc"] = datetime.utcnow().isoformat()
        self._save_manifest(project_id, manifest)
        logger.info(
            "Run '%s' recorded in project '%s'.", run_id, project_id
        )

    def record_failure(
        self,
        project_id: str,
        aoi_name: str,
        error_message: str,
    ) -> None:
        """Record a failed pipeline run in the project manifest.

        Args:
            project_id: Project identifier.
            aoi_name: AOI that failed.
            error_message: Error description.
        """
        try:
            manifest = self.load_project(project_id)
        except GeoAIPlatformError:
            return  # Can't record failure if project doesn't exist.

        manifest["failed_runs"].append({
            "aoi_name": aoi_name,
            "timestamp_utc": datetime.utcnow().isoformat(),
            "error": error_message,
        })
        manifest["updated_utc"] = datetime.utcnow().isoformat()
        self._save_manifest(project_id, manifest)
        logger.warning(
            "Failure recorded for AOI '%s' in project '%s'.",
            aoi_name, project_id,
        )

    def update_summary(
        self,
        project_id: str,
        summary: Dict[str, Any],
    ) -> None:
        """Update the project-level aggregated summary.

        Args:
            project_id: Project identifier.
            summary: Summary dictionary to store.
        """
        try:
            manifest = self.load_project(project_id)
        except GeoAIPlatformError:
            return
        manifest["summary"] = summary
        manifest["updated_utc"] = datetime.utcnow().isoformat()
        self._save_manifest(project_id, manifest)
        logger.debug("Summary updated for project '%s'.", project_id)

    def get_run_history(self, project_id: str) -> List[Dict[str, Any]]:
        """Return the run history for a project.

        Args:
            project_id: Project identifier.

        Returns:
            List of run record dictionaries, most recent first.
        """
        manifest = self.load_project(project_id)
        runs = manifest.get("runs", [])
        return sorted(runs, key=lambda r: r.get("timestamp_utc", ""), reverse=True)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _manifest_path(self, project_id: str) -> Path:
        """Return the manifest file path for a project."""
        return self.manifest_dir / f"{project_id}.json"

    def _save_manifest(self, project_id: str, manifest: Dict[str, Any]) -> None:
        """Write a manifest dictionary to disk."""
        manifest_path = self._manifest_path(project_id)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)
