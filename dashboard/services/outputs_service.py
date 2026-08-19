import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class OutputService:
    """Helpers for inspecting pipeline output directories and artifacts."""

    def __init__(self, output_base_dir: str = "outputs") -> None:
        self.output_base_dir = Path(output_base_dir)

    def list_runs(self, include_stage2: bool = True) -> List[str]:
        if not self.output_base_dir.exists():
            return []
        reserved = {"GeoAI_Platform_Submission", "logs", "projects"}
        run_dirs = [
            p
            for p in self.output_base_dir.iterdir()
            if p.is_dir()
            and p.name not in reserved
            and (include_stage2 or not self._is_stage2_run(p.name))
            and self._has_export_files(p)
        ]
        run_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return [path.name for path in run_dirs]

    @staticmethod
    def _is_stage2_run(run_id: str) -> bool:
        run_name = run_id.lower()
        return "stage2" in run_name or run_name.endswith("_s2")

    @staticmethod
    def _has_export_files(run_dir: Path) -> bool:
        export_dir = run_dir / "exports"
        return export_dir.is_dir() and any(path.is_file() for path in export_dir.iterdir())

    def get_run_dir(self, run_id: str) -> Path:
        return self.output_base_dir / run_id

    def get_export_dir(self, run_id: str) -> Path:
        return self.get_run_dir(run_id) / "exports"

    def list_export_files(self, run_id: str) -> List[Path]:
        export_dir = self.get_export_dir(run_id)
        if not export_dir.exists():
            return []
        return sorted(path for path in export_dir.iterdir() if path.is_file())

    def list_figures(self, run_id: str) -> List[Path]:
        fig_dir = self.get_run_dir(run_id) / "figures"
        if not fig_dir.exists():
            return []
        return sorted(path for path in fig_dir.iterdir() if path.is_file())

    def list_reports(self, run_id: str) -> List[Path]:
        report_dir = self.get_run_dir(run_id) / "reports"
        if not report_dir.exists():
            return []
        return sorted(path for path in report_dir.iterdir() if path.is_file())


    def find_export_file(self, run_id: str, suffix: str) -> Optional[Path]:
        for path in self.list_export_files(run_id):
            if path.name.endswith(suffix):
                return path
        return None

    def find_geojson(self, run_id: str, aoi_name: Optional[str] = None) -> Optional[Path]:
        if aoi_name:
            candidates = [
                f"Classified_Objects_{aoi_name}.geojson",
                f"Change_Objects_{aoi_name}.geojson",
            ]
        else:
            candidates = [p.name for p in self.list_export_files(run_id) if p.suffix == ".geojson"]
        for candidate in candidates:
            path = self.get_export_dir(run_id) / candidate
            if path.exists():
                return path
        geojson_files = [p for p in self.list_export_files(run_id) if p.suffix == ".geojson"]
        return geojson_files[0] if geojson_files else None

    def find_csv(self, run_id: str, aoi_name: Optional[str] = None) -> Optional[Path]:
        export_dir = self.get_export_dir(run_id)
        if not export_dir.exists():
            return None
        if aoi_name:
            candidates = [
                export_dir / f"Object_Statistics_{aoi_name}.csv",
                export_dir / f"Objects_{aoi_name}.csv",
            ]
            return next((path for path in candidates if path.is_file()), None)

        for pattern in ("Object_Statistics_*.csv", "Objects_*.csv"):
            matches = sorted(export_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def read_json(self, path: Path) -> Any:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def read_text(self, path: Path) -> Optional[str]:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def read_binary(self, path: Path) -> Optional[bytes]:
        if not path.exists():
            return None
        return path.read_bytes()

    def get_root_files(self) -> List[Path]:
        if not self.output_base_dir.exists():
            return []
        return [p for p in self.output_base_dir.iterdir() if p.is_file()]

    def get_comparison_data(self) -> Optional[Dict[str, Any]]:
        path = self.output_base_dir / "COMPARISON.json"
        if not path.exists():
            return None
        return self.read_json(path)

    def get_comparison_image(self) -> Optional[Path]:
        path = self.output_base_dir / "comparison_masks.png"
        return path if path.exists() else None

    def get_run_size(self, run_id: str) -> int:
        run_dir = self.get_run_dir(run_id)
        if not run_dir.exists():
            return 0
        return sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
