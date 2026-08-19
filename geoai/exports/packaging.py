"""
PS10 submission packaging for the GeoAI Platform.

Assembles the required output files into a PS10-compliant submission zip
archive and generates the MD5 model hash file required for online and offline
evaluation.

PS10 submission zip naming convention:
    PS10_{DD-MMM-YYYY}_{TeamName}.zip

Required contents per image pair:
    Change_Mask_{Lat}_{Long}.tif   (binary change mask GeoTIFF)
    Change_Mask_{Lat}_{Long}.shp   (and associated .dbf, .shx, .prj)

MD5 hash file:
    {model_name}_md5.txt

Single responsibility: package PS10 submission deliverables.

Position in dependency hierarchy: exports (depends on utils/hash_utils, utils/geo).
"""

import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from geoai.core.exceptions import ExportError
from geoai.utils.constants import PS10_ZIP_PREFIX
from geoai.utils.hash_utils import compute_and_save_model_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PS10 submission zip assembly
# ---------------------------------------------------------------------------


def build_ps10_submission_zip(
    result_files: List[Union[str, Path]],
    output_dir: Union[str, Path],
    team_name: str,
    submission_date: Optional[datetime] = None,
) -> Path:
    """Assemble a PS10-compliant submission zip archive.

    Collects all result files (GeoTIFFs, shapefiles, and associated files)
    into a zip archive named according to the PS10 convention:
        PS10_{DD-MMM-YYYY}_{TeamName}.zip

    All files are placed at the root level of the zip archive (no
    subdirectory nesting), as required by the PS10 submission format.

    Args:
        result_files: List of file paths to include in the zip. Should
            include the GeoTIFF change masks and all shapefile components
            (.shp, .dbf, .shx, .prj, .cpg).
        output_dir: Directory where the zip file will be written.
        team_name: Team or startup name (no spaces — spaces are replaced
            with underscores). Used in the zip filename.
        submission_date: Date to include in the filename. Defaults to today.

    Returns:
        Resolved absolute path of the written zip file.

    Raises:
        ExportError: If the zip file cannot be created or any source file
            is missing.
    """
    if submission_date is None:
        submission_date = datetime.utcnow()

    date_str = submission_date.strftime("%d-%b-%Y")
    clean_team_name = team_name.replace(" ", "_")
    zip_name = f"{PS10_ZIP_PREFIX}_{date_str}_{clean_team_name}.zip"
    zip_path = Path(output_dir) / zip_name
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building PS10 submission zip: '%s' (%d files).",
        zip_name, len(result_files),
    )

    # Expand shapefile wildcards — include all associated component files.
    expanded_files = _expand_shapefile_components(result_files)

    missing = [f for f in expanded_files if not Path(f).exists()]
    if missing:
        raise ExportError(
            f"Cannot build submission zip — {len(missing)} source file(s) "
            f"are missing: {[str(f) for f in missing]}."
        )

    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in expanded_files:
                file_path = Path(file_path)
                zf.write(file_path, arcname=file_path.name)
                logger.debug("Added to zip: '%s'.", file_path.name)
    except Exception as exc:
        raise ExportError(
            f"Failed to create submission zip '{zip_path}': {exc}."
        ) from exc

    size_mb = zip_path.stat().st_size / (1024 ** 2)
    logger.info(
        "PS10 submission zip created: '%s' (%.2f MB, %d files).",
        zip_name, size_mb, len(expanded_files),
    )
    return zip_path.resolve()


# ---------------------------------------------------------------------------
# MD5 hash generation for PS10 submission
# ---------------------------------------------------------------------------


def generate_model_hash_for_submission(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
) -> tuple:
    """Compute and save the MD5 hash of the model file for PS10 submission.

    The MD5 hash must be submitted alongside the online results on 31 October
    2025 and is verified during offline evaluation.

    Args:
        model_path: Path to the model pkl file.
        output_dir: Directory where the hash text file will be written.

    Returns:
        Tuple of (hash_string, hash_file_path) as returned by
        :func:`~geoai.utils.hash_utils.compute_and_save_model_hash`.

    Raises:
        ModelNotFoundError: If the model file does not exist.
        ModelError: If the model cannot be read or hash file cannot be written.
    """
    logger.info(
        "Generating MD5 hash for PS10 submission — model: '%s'.",
        Path(model_path).name,
    )
    return compute_and_save_model_hash(
        model_path=model_path,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _expand_shapefile_components(
    file_paths: List[Union[str, Path]],
) -> List[Path]:
    """Expand shapefile paths to include all required component files.

    A shapefile consists of multiple files with the same stem but different
    extensions (.shp, .dbf, .shx, .prj, optionally .cpg, .sbn, .sbx).
    This function finds all existing component files for each .shp in the
    input list.

    Args:
        file_paths: List of file paths. Non-shapefile paths are included
            as-is. Each .shp path is expanded to include all components.

    Returns:
        Expanded list of paths with all shapefile components included.
    """
    shapefile_extensions = {".shp", ".dbf", ".shx", ".prj", ".cpg", ".sbn", ".sbx"}
    result = []
    seen = set()

    for path in file_paths:
        path = Path(path)
        if path.suffix.lower() == ".shp":
            # Add all component files that exist.
            for ext in shapefile_extensions:
                component = path.with_suffix(ext)
                if component.exists() and str(component) not in seen:
                    result.append(component)
                    seen.add(str(component))
        else:
            if str(path) not in seen:
                result.append(path)
                seen.add(str(path))

    return result
