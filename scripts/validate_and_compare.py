#!/usr/bin/env python
"""
Comprehensive Validation & Comparison Script
=============================================

Validates export quality, compares V1 vs V2 results, and generates
submission-ready deliverables with comprehensive reports.

Usage:
    python scripts/validate_and_compare.py --runs PS10_20260623_124542 Dholera_20260623_124556 --generate-report --create-submission

Author: GeoAI Platform v2
Date: June 2026
"""

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

try:
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    from matplotlib import pyplot as plt
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from geoai.core.logger import get_run_logger

logger = get_run_logger(__name__)

class ValidationReport:
    """Generate comprehensive validation reports."""
    
    def __init__(self):
        self.sections = []
        self.metadata = {
            "timestamp": datetime.now().isoformat(),
            "version": "2.0",
            "checks_passed": 0,
            "checks_failed": 0,
            "warnings": []
        }
    
    def add_section(self, title, content):
        """Add section to report."""
        self.sections.append({"title": title, "content": content})
    
    def check_file(self, path, expected_size_mb=None):
        """Validate file exists and has expected size."""
        p = Path(path)
        if not p.exists():
            self.metadata["checks_failed"] += 1
            return f"❌ MISSING: {path}"
        
        size_mb = p.stat().st_size / (1024 ** 2)
        self.metadata["checks_passed"] += 1
        
        status = f"✓ EXISTS: {path} ({size_mb:.2f} MB)"
        if expected_size_mb and size_mb < expected_size_mb:
            self.metadata["warnings"].append(f"Small file size: {path}")
            status += " ⚠ (small)"
        
        return status
    
    def check_shapefile_integrity(self, shp_path):
        """Validate shapefile attributes and geometry."""
        checks = []
        try:
            gdf = gpd.read_file(shp_path)
            checks.append(f"✓ Shapefile readable: {len(gdf)} features")
            
            # Check geometry validity
            valid_geoms = gdf.geometry.is_valid.sum()
            total_geoms = len(gdf)
            checks.append(f"✓ Geometry valid: {valid_geoms}/{total_geoms}")
            
            # Check CRS
            if gdf.crs:
                checks.append(f"✓ CRS defined: {gdf.crs}")
            else:
                checks.append("⚠ CRS not defined")
            
            # Check attributes
            if len(gdf.columns) > 0:
                checks.append(f"✓ Attributes: {', '.join(gdf.columns[:5])}...")
            
            self.metadata["checks_passed"] += len(checks)
        except Exception as e:
            checks.append(f"❌ Error reading: {e}")
            self.metadata["checks_failed"] += 1
        
        return checks
    
    def check_geojson_validity(self, geojson_path):
        """Validate GeoJSON structure."""
        checks = []
        try:
            with open(geojson_path, 'r') as f:
                data = json.load(f)
            
            if "type" in data and data["type"] == "FeatureCollection":
                checks.append(f"✓ Valid GeoJSON: {len(data['features'])} features")
                self.metadata["checks_passed"] += 1
            else:
                checks.append("❌ Invalid GeoJSON structure")
                self.metadata["checks_failed"] += 1
            
            if data.get("features"):
                first_props = data["features"][0].get("properties", {})
                if first_props:
                    checks.append(f"✓ Properties present: {list(first_props.keys())[:3]}...")
        except Exception as e:
            checks.append(f"❌ Error parsing: {e}")
            self.metadata["checks_failed"] += 1
        
        return checks
    
    def check_csv_data(self, csv_path):
        """Validate CSV data integrity."""
        checks = []
        try:
            df = pd.read_csv(csv_path)
            checks.append(f"✓ CSV readable: {len(df)} rows, {len(df.columns)} columns")
            
            if len(df) > 0:
                checks.append(f"✓ Data populated: first row has {df.iloc[0].notna().sum()} non-null values")
            
            self.metadata["checks_passed"] += len(checks)
        except Exception as e:
            checks.append(f"❌ Error reading: {e}")
            self.metadata["checks_failed"] += 1
        
        return checks
    
    def to_markdown(self):
        """Export report as markdown."""
        lines = [
            "# GeoAI Platform v2 - Validation Report",
            f"\n**Generated**: {self.metadata['timestamp']}",
            f"**Checks Passed**: {self.metadata['checks_passed']} ✓",
            f"**Checks Failed**: {self.metadata['checks_failed']} ❌",
        ]
        
        if self.metadata["warnings"]:
            lines.append(f"\n**Warnings**: {len(self.metadata['warnings'])}")
            for w in self.metadata["warnings"]:
                lines.append(f"- {w}")
        
        for section in self.sections:
            lines.append(f"\n## {section['title']}")
            if isinstance(section['content'], list):
                for item in section['content']:
                    lines.append(f"  {item}")
            else:
                lines.append(section['content'])
        
        return "\n".join(lines)


def validate_exports(run_id):
    """Validate all export files for a run."""
    report = ValidationReport()
    base_dir = Path(f"outputs/{run_id}/exports")
    
    logger.info(f"Validating exports in: {base_dir}")
    
    if not base_dir.exists():
        report.add_section("Export Directory", f"❌ Directory not found: {base_dir}")
        return report
    
    # Expected files
    expected_files = [
        (f"Change_Mask_*.tif", "GeoTIFF change mask"),
        (f"Change_Mask_*.shp", "Shapefile change polygons"),
        (f"Change_Objects_*.geojson", "GeoJSON object features"),
        (f"Object_Statistics_*.csv", "CSV object statistics"),
    ]
    
    found_files = defaultdict(list)
    for pattern, desc in expected_files:
        matches = list(base_dir.glob(pattern))
        if matches:
            found_files[desc] = matches
    
    checks = []
    for desc, files in found_files.items():
        if files:
            checks.append(f"✓ {desc}: {len(files)} file(s)")
            for f in files:
                checks.append(f"  → {f.name}")
    
    report.add_section("Export Files Found", checks)
    
    # Validate specific formats
    tif_files = list(base_dir.glob("Change_Mask_*.tif"))
    if tif_files:
        tif_checks = []
        for tif in tif_files:
            try:
                with rasterio.open(tif) as src:
                    shape = src.shape
                    dtype = src.dtypes[0]
                    crs = src.crs
                    tif_checks.append(
                        f"✓ {tif.name}: shape={shape} dtype={dtype} crs={crs}"
                    )
            except Exception as e:
                tif_checks.append(f"❌ {tif.name}: {e}")
        report.add_section("GeoTIFF Validation", tif_checks)
    
    # Validate shapefiles
    shp_files = list(base_dir.glob("*.shp"))
    for shp in shp_files:
        shp_checks = report.check_shapefile_integrity(str(shp))
        report.add_section(f"Shapefile: {shp.name}", shp_checks)
    
    # Validate GeoJSON
    geojson_files = list(base_dir.glob("*.geojson"))
    for geojson in geojson_files:
        gj_checks = report.check_geojson_validity(str(geojson))
        report.add_section(f"GeoJSON: {geojson.name}", gj_checks)
    
    # Validate CSVs
    csv_files = list(base_dir.glob("*.csv"))
    for csv in csv_files:
        csv_checks = report.check_csv_data(str(csv))
        report.add_section(f"CSV: {csv.name}", csv_checks)
    
    return report


def compare_results(run_ids):
    """Compare results across multiple runs."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "runs": {}
    }
    
    logger.info(f"Comparing {len(run_ids)} runs...")
    
    for run_id in run_ids:
        exports_dir = Path(f"outputs/{run_id}/exports")
        stats_files = list(exports_dir.glob("Object_Statistics_*.csv"))
        
        if not stats_files:
            # Fallback to any CSV file
            stats_files = list(exports_dir.glob("*.csv"))
        
        stats_file = stats_files[0] if stats_files else None
        
        if not stats_file or not stats_file.exists():
            logger.warning(f"No statistics file found for {run_id}")
            continue
        
        try:
            df = pd.read_csv(stats_file)
            comparison["runs"][run_id] = {
                "total_objects": len(df),
                "total_area_ha": df["area_ha"].sum() if "area_ha" in df.columns else 0,
                "mean_area_ha": df["area_ha"].mean() if "area_ha" in df.columns else 0,
                "max_area_ha": df["area_ha"].max() if "area_ha" in df.columns else 0,
            }
        except Exception as e:
            logger.error(f"Error reading {stats_file}: {e}")
    
    return comparison


def generate_visualizations(run_ids):
    """Generate comparison visualizations."""
    logger.info(f"Generating visualizations for {len(run_ids)} runs...")
    
    fig, axes = plt.subplots(1, len(run_ids), figsize=(15, 6))
    if len(run_ids) == 1:
        axes = [axes]
    
    for idx, run_id in enumerate(run_ids):
        try:
            # Load change mask
            tif_file = next(Path(f"outputs/{run_id}/exports").glob("Change_Mask_*.tif"))
            with rasterio.open(tif_file) as src:
                mask = src.read(1)
            
            ax = axes[idx]
            im = ax.imshow(mask, cmap="RdYlGn_r", vmin=0, vmax=1)
            ax.set_title(f"{run_id.split('_')[0]} Change Mask")
            ax.axis("off")
            plt.colorbar(im, ax=ax, label="Change (0=No, 1=Yes)")
        
        except Exception as e:
            logger.warning(f"Could not visualize {run_id}: {e}")
    
    output_file = "outputs/comparison_masks.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Visualization saved: {output_file}")
    plt.close()


def create_submission_package(run_ids, output_name="GeoAI_Platform_Submission"):
    """Create final submission package with all artifacts."""
    logger.info(f"Creating submission package: {output_name}")
    
    submission_dir = Path(f"outputs/{output_name}")
    submission_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (submission_dir / "aoi_results").mkdir(exist_ok=True)
    (submission_dir / "exports").mkdir(exist_ok=True)
    (submission_dir / "reports").mkdir(exist_ok=True)
    
    # Copy exports from each run
    for run_id in run_ids:
        aoi_name = run_id.split("_")[0]
        run_dir = Path(f"outputs/{run_id}/exports")
        
        if run_dir.exists():
            dest_dir = submission_dir / "aoi_results" / aoi_name
            dest_dir.mkdir(exist_ok=True)
            
            for file in run_dir.glob("*"):
                if file.is_file():
                    shutil.copy2(file, dest_dir / file.name)
            
            logger.info(f"  Copied {aoi_name} exports: {aoi_name}")
    
    # Generate summary report
    summary = {
        "title": "GeoAI Platform v2 - Execution Summary",
        "generated": datetime.now().isoformat(),
        "runs": run_ids,
        "aois": list(set([rid.split("_")[0] for rid in run_ids])),
        "results": {}
    }
    
    for run_id in run_ids:
        exports_dir = Path(f"outputs/{run_id}/exports")
        stats_files = list(exports_dir.glob("Object_Statistics_*.csv"))
        if not stats_files:
            stats_files = list(exports_dir.glob("*.csv"))
        stats_file = stats_files[0] if stats_files else None
        
        if stats_file and stats_file.exists():
            df = pd.read_csv(stats_file)
            summary["results"][run_id] = {
                "objects": len(df),
                "area_ha": float(df["area_ha"].sum()) if "area_ha" in df.columns else 0,
            }
    
    # Write summary
    summary_file = submission_dir / "SUMMARY.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✓ Submission package created: {submission_dir}")
    logger.info(f"  Total size: {sum(f.stat().st_size for f in submission_dir.rglob('*') if f.is_file()) / (1024**2):.1f} MB")
    
    return submission_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate exports and compare results"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run IDs to validate (e.g., PS10_20260623_124542 Dholera_20260623_124556)"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate validation reports"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare results across runs"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate comparison visualizations"
    )
    parser.add_argument(
        "--create-submission",
        action="store_true",
        help="Create final submission package"
    )
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("GeoAI Platform v2 - Validation & Comparison")
    logger.info("="*70)
    
    # Validation
    if args.generate_report:
        for run_id in args.runs:
            logger.info(f"\nValidating: {run_id}")
            report = validate_exports(run_id)
            
            report_file = Path(f"outputs/{run_id}/VALIDATION_REPORT.md")
            report_file.write_text(report.to_markdown(), encoding="utf-8")
            logger.info(f"Report saved: {report_file}")
    
    # Comparison
    if args.compare:
        comparison = compare_results(args.runs)
        comp_file = Path("outputs/COMPARISON.json")
        with open(comp_file, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison saved: {comp_file}")
    
    # Visualization
    if args.visualize:
        try:
            generate_visualizations(args.runs)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    # Submission
    if args.create_submission:
        submission_dir = create_submission_package(args.runs)
        logger.info(f"\n✓ SUBMISSION READY: {submission_dir}")
    
    logger.info("="*70)
    logger.info("✓ Validation complete")
    logger.info("="*70)


if __name__ == "__main__":
    main()
