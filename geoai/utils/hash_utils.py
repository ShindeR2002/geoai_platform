"""
MD5 hash utilities for the GeoAI Platform.

Provides functions for computing and verifying MD5 checksums of model files.
The PS10 challenge requires participants to submit an MD5 hash of their model
file alongside their results, and to present the model for hash verification
during offline evaluation.

Single responsibility: compute, verify, and persist MD5 checksums.

Position in dependency hierarchy: utils (no internal imports).
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Union

from geoai.core.exceptions import ModelNotFoundError, ModelError

logger = logging.getLogger(__name__)

# Block size for streaming file reads. 64 KB balances memory use and I/O
# efficiency for model files in the 100 MB range.
_READ_BLOCK_SIZE: int = 65_536


# ---------------------------------------------------------------------------
# Hash computation
# ---------------------------------------------------------------------------


def compute_file_md5(path: Union[str, Path]) -> str:
    """Compute the MD5 checksum of a file using streaming reads.

    Reads the file in blocks to avoid loading the entire file into memory,
    which is important for large model files (e.g., rf_enhanced.pkl at 113 MB).

    Args:
        path: Path to the file to hash.

    Returns:
        Lowercase hexadecimal MD5 digest string (32 characters).

    Raises:
        ModelNotFoundError: If the file does not exist at ``path``.
        ModelError: If the file cannot be read.
    """
    path = Path(path)
    if not path.exists():
        raise ModelNotFoundError(str(path))

    logger.info("Computing MD5 hash for '%s'.", path)

    hasher = hashlib.md5()
    try:
        with open(path, "rb") as fh:
            while True:
                block = fh.read(_READ_BLOCK_SIZE)
                if not block:
                    break
                hasher.update(block)
    except OSError as exc:
        raise ModelError(
            f"Failed to read file '{path}' for MD5 hashing: {exc}."
        ) from exc

    digest = hasher.hexdigest()
    logger.info("MD5 hash for '%s': %s", path.name, digest)
    return digest


def verify_file_md5(path: Union[str, Path], expected_hash: str) -> bool:
    """Verify that a file's MD5 hash matches an expected value.

    Used during offline evaluation to confirm that the submitted model file
    is identical to the one used to generate the online submission results.

    Args:
        path: Path to the file to verify.
        expected_hash: Expected MD5 digest string (case-insensitive).

    Returns:
        True if the computed hash matches ``expected_hash``; False otherwise.

    Raises:
        ModelNotFoundError: If the file does not exist.
        ModelError: If the file cannot be read.
    """
    actual_hash = compute_file_md5(path)
    match = actual_hash.lower() == expected_hash.lower().strip()
    if match:
        logger.info(
            "MD5 verification PASSED for '%s': %s.", Path(path).name, actual_hash
        )
    else:
        logger.warning(
            "MD5 verification FAILED for '%s'. "
            "Expected: %s  Actual: %s",
            Path(path).name,
            expected_hash.lower(),
            actual_hash,
        )
    return match


# ---------------------------------------------------------------------------
# Hash persistence
# ---------------------------------------------------------------------------


def save_hash_to_file(
    file_hash: str,
    source_path: Union[str, Path],
    output_path: Union[str, Path],
) -> Path:
    """Save an MD5 hash to a text file in PS10 submission format.

    The output file contains the hash and the source filename on a single
    line, conforming to the standard ``md5sum`` output format.

    Args:
        file_hash: MD5 digest string to save.
        source_path: Path to the file that was hashed (used for the filename
            in the output).
        output_path: Path where the hash text file should be written.

    Returns:
        The resolved absolute path of the written hash file.

    Raises:
        ModelError: If the hash file cannot be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_name = Path(source_path).name
    content = f"{file_hash}  {source_name}\n"
    try:
        output_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ModelError(
            f"Failed to write hash file to '{output_path}': {exc}."
        ) from exc

    logger.info("MD5 hash saved to '%s'.", output_path)
    return output_path.resolve()


def compute_and_save_model_hash(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    output_filename: Optional[str] = None,
) -> tuple:
    """Compute and persist the MD5 hash of a model file.

    Convenience function that combines :func:`compute_file_md5` and
    :func:`save_hash_to_file` for the PS10 submission workflow.

    Args:
        model_path: Path to the model file (e.g. ``models/rf/rf_enhanced.pkl``).
        output_dir: Directory where the hash text file should be written.
        output_filename: Name for the hash file. If None, defaults to
            ``'{model_name}_md5.txt'``.

    Returns:
        Tuple of ``(hash_string, hash_file_path)`` where ``hash_string`` is
        the 32-character hex digest and ``hash_file_path`` is the absolute
        path of the written text file.

    Raises:
        ModelNotFoundError: If the model file does not exist.
        ModelError: If the model cannot be read or the hash file cannot be written.
    """
    model_path = Path(model_path)
    model_hash = compute_file_md5(model_path)

    if output_filename is None:
        output_filename = f"{model_path.stem}_md5.txt"

    hash_file_path = save_hash_to_file(
        file_hash=model_hash,
        source_path=model_path,
        output_path=Path(output_dir) / output_filename,
    )
    return model_hash, hash_file_path


def load_hash_from_file(hash_file_path: Union[str, Path]) -> str:
    """Read an MD5 hash from a text file written by :func:`save_hash_to_file`.

    Args:
        hash_file_path: Path to the hash text file.

    Returns:
        The MD5 digest string extracted from the file.

    Raises:
        ModelError: If the file cannot be read or does not contain a valid
            MD5 hash on the first line.
    """
    hash_file_path = Path(hash_file_path)
    if not hash_file_path.exists():
        raise ModelError(
            f"Hash file not found: '{hash_file_path}'."
        )
    try:
        first_line = hash_file_path.read_text(encoding="utf-8").splitlines()[0]
        hash_part = first_line.split()[0]
    except (IndexError, OSError) as exc:
        raise ModelError(
            f"Could not parse MD5 hash from '{hash_file_path}': {exc}."
        ) from exc

    if len(hash_part) != 32 or not all(c in "0123456789abcdef" for c in hash_part.lower()):
        raise ModelError(
            f"Value '{hash_part}' in '{hash_file_path}' does not appear to be "
            "a valid MD5 hex digest."
        )

    return hash_part.lower()
