"""
Dashboard report generator for the GeoAI Platform.

Generates structured project-level reports aggregating statistics across
multiple AOI runs for dashboard and API consumption.

Single responsibility: generate project-level reports from run history.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from geoai.dashboard.project_manager import ProjectManager

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates automated project reports from run history.

    Args:
        project_manager: ProjectManager instance for accessing run history.
        output_dir: Directory where generated reports are written.
    """

    def __init__(
        self,
        project_manager: ProjectManager,
        output_dir: str = "outputs/reports",
    ) -> None:
        self.project_manager = project_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_project_report(
        self,
        project_id: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a Markdown project report from run history.

        Args:
            project_id: Project identifier.
            output_path: Optional file path to write the report. Auto-generated
                if None.

        Returns:
            Report content as a Markdown string.
        """
        manifest = self.project_manager.load_project(project_id)
        runs = manifest.get("runs", [])
        summary = manifest.get("summary", {})
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"# GeoAI Platform Project Report",
            f"",
            f"**Project ID:** {project_id}  ",
            f"**Generated:** {now}  ",
            f"**Description:** {manifest.get('description', 'N/A')}",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| AOIs Processed | {summary.get('total_aois_processed', len(runs))} |",
            f"| Total Change Area | {summary.get('total_change_area_ha', 0):.4f} ha |",
            f"| Total Objects | {summary.get('total_significant_objects', 0)} |",
            f"",
            f"## Run History",
            f"",
            f"| Run ID | AOI | Change Area (ha) | Objects | Timestamp |",
            f"|--------|-----|-----------------|---------|-----------|",
        ]

        for run in runs:
            lines.append(
                f"| {run.get('run_id', '')} "
                f"| {run.get('aoi_name', '')} "
                f"| {run.get('change_area_ha', 0):.4f} "
                f"| {run.get('object_count', 0)} "
                f"| {run.get('timestamp_utc', '')[:19]} |"
            )

        if manifest.get("failed_runs"):
            lines += [
                f"",
                f"## Failed Runs",
                f"",
            ]
            for fail in manifest["failed_runs"]:
                lines.append(
                    f"- **{fail.get('aoi_name', 'Unknown')}** "
                    f"({fail.get('timestamp_utc', '')[:19]}): "
                    f"{fail.get('error', 'Unknown error')}"
                )

        report = "\n".join(lines)

        if output_path is None:
            output_path = self.output_dir / f"{project_id}_report.md"
        Path(output_path).write_text(report, encoding="utf-8")
        logger.info("Project report written: '%s'.", output_path)
        return report
