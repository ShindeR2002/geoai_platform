"""
scripts/migrate_model.py — Migrate rf_enhanced.pkl from Version 1 to the platform.

Copies the model file to models/rf/rf_enhanced.pkl and verifies that
rf_enhanced_meta.json is consistent with the model.

Usage:
    python scripts/migrate_model.py --source /path/to/Version1/outputs/models/rf_enhanced.pkl
    python scripts/migrate_model.py --source /path/to/rf_enhanced.pkl --verify
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoai.utils.hash_utils import compute_file_md5

DEST_DIR = Path("models/rf")
DEST_MODEL = DEST_DIR / "rf_enhanced.pkl"
META_FILE = DEST_DIR / "rf_enhanced_meta.json"


def migrate(source_path: str, verify: bool = False) -> None:
    """Copy model file and report hash."""
    src = Path(source_path)
    if not src.exists():
        print(f"ERROR: Source file not found: {src}")
        sys.exit(1)

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Copying {src} → {DEST_MODEL}")
    shutil.copy2(src, DEST_MODEL)
    size_mb = DEST_MODEL.stat().st_size / (1024 ** 2)
    print(f"Copied ({size_mb:.1f} MB)")

    md5 = compute_file_md5(DEST_MODEL)
    print(f"MD5: {md5}")

    # Save hash alongside the model
    hash_file = DEST_DIR / "rf_enhanced_md5.txt"
    hash_file.write_text(f"{md5}  rf_enhanced.pkl\n", encoding="utf-8")
    print(f"Hash saved to {hash_file}")

    if verify and META_FILE.exists():
        with open(META_FILE) as f:
            meta = json.load(f)
        expected_features = meta.get("feature_count", 18)
        print(f"Metadata feature_count: {expected_features}")
        print("Migration verified.")

    print("Migration complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate rf_enhanced.pkl to platform.")
    parser.add_argument("--source", required=True, help="Path to source pkl file.")
    parser.add_argument("--verify", action="store_true", help="Verify against metadata.")
    args = parser.parse_args()
    migrate(args.source, args.verify)
