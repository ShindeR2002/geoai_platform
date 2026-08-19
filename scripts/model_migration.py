#!/usr/bin/env python
"""
Migration Script: Extract and Validate RF Enhanced Model from Version 1
=========================================================================

This script extracts the trained Random Forest model from Version 1 and
validates its compatibility with the Version 2 platform architecture.

Usage:
    python scripts/model_migration.py --v1-path /path/to/Version1/outputs/models/rf_enhanced.pkl --verify

Author: GeoAI Platform v2
Date: June 2026
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install scikit-learn joblib numpy")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geoai.utils.constants import CANONICAL_FEATURE_COUNT, CANONICAL_FEATURE_NAMES
from geoai.utils.hash_utils import compute_file_md5

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)

DEST_DIR = Path("models/rf")
DEST_MODEL = DEST_DIR / "rf_enhanced.pkl"
META_FILE = DEST_DIR / "rf_enhanced_meta.json"


def validate_model_structure(model):
    """Validate the loaded model has expected structure."""
    errors = []
    
    # Check it's a RandomForestClassifier
    if not isinstance(model, RandomForestClassifier):
        errors.append(f"Model is {type(model)}, expected RandomForestClassifier")
    
    # Check n_estimators
    if hasattr(model, 'n_estimators') and model.n_estimators != 100:
        logger.warning(
            f"Expected n_estimators=100, found {model.n_estimators}. "
            "This may affect inference results."
        )
    
    # Check n_features_in_
    if hasattr(model, 'n_features_in_'):
        if model.n_features_in_ != CANONICAL_FEATURE_COUNT:
            errors.append(
                f"Model trained on {model.n_features_in_} features, "
                f"platform expects {CANONICAL_FEATURE_COUNT}"
            )
    
    # Check classes (binary classification)
    if hasattr(model, 'classes_'):
        if not np.array_equal(model.classes_, np.array([0, 1])):
            errors.append(
                f"Model classes are {model.classes_}, expected [0, 1]"
            )
    
    # Check feature_importances_
    if not hasattr(model, 'feature_importances_'):
        errors.append("Model missing feature_importances_ attribute")
    elif len(model.feature_importances_) != CANONICAL_FEATURE_COUNT:
        errors.append(
            f"Feature importances has {len(model.feature_importances_)} values, "
            f"expected {CANONICAL_FEATURE_COUNT}"
        )
    
    return errors


def create_metadata(model, model_path):
    """Create metadata JSON file for the model."""
    metadata = {
        "model_id": "rf_enhanced_v1",
        "feature_count": CANONICAL_FEATURE_COUNT,
        "feature_names": list(CANONICAL_FEATURE_NAMES),
        "model_class": "RandomForestClassifier",
        "n_estimators": model.n_estimators if hasattr(model, 'n_estimators') else None,
        "n_features_in": model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
        "classes": model.classes_.tolist() if hasattr(model, 'classes_') else None,
        "source": "Version 1 Notebook 05",
        "migration_date": "2026-06-23",
        "model_file_md5": compute_file_md5(model_path)
    }
    
    return metadata


def migrate_model(v1_path, verify=False):
    """Migrate and validate RF Enhanced model from Version 1 to Version 2."""
    
    v1_path = Path(v1_path)
    
    # Step 1: Validate source file exists
    logger.info(f"Validating source file: {v1_path}")
    if not v1_path.exists():
        logger.error(f"Source file not found: {v1_path}")
        return False
    
    # Step 2: Load and validate model structure
    logger.info("Loading model from Version 1...")
    try:
        rf = joblib.load(v1_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Step 3: Validate model structure
    logger.info("Validating model structure...")
    validation_errors = validate_model_structure(rf)
    
    if validation_errors:
        logger.warning("⚠ Validation warnings:")
        for error in validation_errors:
            logger.warning(f"  - {error}")
    else:
        logger.info("✓ Model structure valid")
    
    # Step 4: Copy model file
    logger.info(f"Creating destination directory: {DEST_DIR}")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Copying model: {v1_path} → {DEST_MODEL}")
    try:
        shutil.copy2(v1_path, DEST_MODEL)
    except Exception as e:
        logger.error(f"Failed to copy model: {e}")
        return False
    
    size_mb = DEST_MODEL.stat().st_size / (1024 ** 2)
    logger.info(f"✓ Model copied ({size_mb:.1f} MB)")
    
    # Step 5: Create metadata
    logger.info("Creating metadata file...")
    metadata = create_metadata(rf, DEST_MODEL)
    
    try:
        with open(META_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Metadata saved to {META_FILE}")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        return False
    
    # Step 6: Save model hash
    md5 = compute_file_md5(DEST_MODEL)
    hash_file = DEST_DIR / "rf_enhanced_md5.txt"
    hash_file.write_text(f"{md5}  rf_enhanced.pkl\n", encoding="utf-8")
    logger.info(f"✓ Model MD5: {md5}")
    
    # Step 7: Optional verification with platform loader
    if verify:
        logger.info("Verifying model with platform loader...")
        try:
            from geoai.models.registry import load_model
            
            loaded_model = load_model(
                model_id="rf_enhanced_v1",
                model_path=DEST_MODEL,
                metadata_path=META_FILE
            )
            
            logger.info(f"✓ Platform verification successful")
            logger.info(f"  - Model ID: {loaded_model.model_id}")
            logger.info(f"  - Feature count: {loaded_model.get_expected_feature_count()}")
            logger.info(f"  - Feature names: {', '.join(loaded_model.get_feature_names()[:3])}...")
            
        except Exception as e:
            logger.error(f"Platform verification failed: {e}")
            return False
    
    # Step 8: Summary
    logger.info("\n" + "="*70)
    logger.info("✓ MIGRATION SUCCESSFUL")
    logger.info("="*70)
    logger.info(f"Model location:    {DEST_MODEL}")
    logger.info(f"Metadata location: {META_FILE}")
    logger.info(f"MD5 hash:          {md5}")
    logger.info("\nNext steps:")
    logger.info("  1. Verify platform can load model:")
    logger.info("     python -c \"from geoai.models.registry import load_model; m = load_model('rf_enhanced_v1', 'models/rf/rf_enhanced.pkl'); print(f'Loaded: {m.model_id}')\"")
    logger.info("\n  2. Run pipeline with production model:")
    logger.info("     python run_pipeline.py --stage 1 --aoi PS10")
    logger.info("\n  3. Check outputs in:")
    logger.info("     outputs/<AOI>_*/exports/")
    logger.info("\n" + "="*70)
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate RF Enhanced model from Version 1 to Version 2 platform"
    )
    parser.add_argument(
        "--v1-path",
        required=True,
        help="Path to Version 1 rf_enhanced.pkl file"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model compatibility with platform loader"
    )
    
    args = parser.parse_args()
    
    success = migrate_model(args.v1_path, verify=args.verify)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
