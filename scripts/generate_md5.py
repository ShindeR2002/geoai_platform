"""
scripts/generate_md5.py — Generate MD5 hash of the model file for PS10 submission.

The PS10 challenge requires participants to submit an MD5 hash of their model
file with their online results (31 October 2025) and to present the model for
hash verification during offline evaluation at IIT Delhi.

Usage:
    python scripts/generate_md5.py
    python scripts/generate_md5.py --model models/rf/rf_enhanced.pkl --output outputs/
    python scripts/generate_md5.py --verify <hash_value>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from geoai.utils.hash_utils import (
    compute_and_save_model_hash,
    compute_file_md5,
    load_hash_from_file,
    verify_file_md5,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify MD5 hash of the GeoAI Platform model."
    )
    parser.add_argument(
        "--model",
        default="models/rf/rf_enhanced.pkl",
        help="Path to the model file. Default: models/rf/rf_enhanced.pkl",
    )
    parser.add_argument(
        "--output",
        default="outputs/",
        help="Directory to write the hash text file. Default: outputs/",
    )
    parser.add_argument(
        "--verify",
        default=None,
        help="Verify the model against this expected hash string.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        print("Run: python scripts/migrate_model.py --source /path/to/rf_enhanced.pkl")
        return 1

    if args.verify:
        # Verification mode
        print(f"Verifying MD5 hash for: {model_path}")
        match = verify_file_md5(model_path, args.verify)
        if match:
            print(f"VERIFIED: Hash matches {args.verify}")
            return 0
        else:
            actual = compute_file_md5(model_path)
            print(f"MISMATCH: Expected {args.verify}, got {actual}")
            return 1

    # Generation mode
    print(f"Computing MD5 hash for: {model_path.name}")
    md5_hash, hash_file = compute_and_save_model_hash(
        model_path=model_path,
        output_dir=args.output,
    )
    print(f"\nMD5 Hash: {md5_hash}")
    print(f"Saved to: {hash_file}")
    print(
        "\nInclude this hash file in your PS10 submission zip "
        "and submit the hash value on the PS10 website."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
