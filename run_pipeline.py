"""
GeoAI Platform — Command-line entry point.

Usage:
    python run_pipeline.py --stage 1 --aoi Dholera
    python run_pipeline.py --stage 2 --aoi Dholera --run-id Dholera_20260620_143012
    python run_pipeline.py --stage all --aoi PS10 Dholera
    python run_pipeline.py --stage 3
    python run_pipeline.py --api
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the project root to sys.path when run directly.
sys.path.insert(0, str(Path(__file__).parent))

from geoai.core.config import load_platform_config
from geoai.core.logger import configure_logging, configure_logging_from_config


def main() -> int:
    """Parse arguments and dispatch to the appropriate pipeline stage.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    parser = argparse.ArgumentParser(
        description="GeoAI Platform — PS10 Anthropogenic Change Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs",
        help="Path to the configs/ directory. Default: configs/",
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "all", "api"],
        required=True,
        help=(
            "Pipeline stage to run: "
            "1=change detection, 2=classification, "
            "3=batch, all=1+2, api=start REST API."
        ),
    )
    parser.add_argument(
        "--aoi",
        nargs="+",
        default=None,
        help="AOI name(s) to process. Defaults to all configured AOIs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stage 1 run ID to use as input for Stage 2.",
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Project identifier for Stage 3 batch runs.",
    )
    parser.add_argument(
        "--compute-shap",
        action="store_true",
        help="Compute SHAP feature attributions during Stage 2.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO.",
    )

    args = parser.parse_args()

    # Configure logging before loading config (config loading logs).
    configure_logging(log_level=args.log_level)
    logger = logging.getLogger("geoai.run_pipeline")

    logger.info("GeoAI Platform starting — stage=%s aoi=%s", args.stage, args.aoi)

    # Load platform configuration.
    try:
        config = load_platform_config(args.config)
    except Exception as exc:
        logger.error("Failed to load configuration from '%s': %s", args.config, exc)
        return 1

    # Re-configure logging with settings from config file (overrides CLI default).
    try:
        import yaml
        log_yaml = Path(args.config) / "logging.yaml"
        if log_yaml.exists():
            with open(log_yaml) as f:
                log_cfg = yaml.safe_load(f)
            if log_cfg:
                configure_logging_from_config(log_cfg)
    except Exception:
        pass  # Continue with CLI log level if config logging fails.

    # Dispatch to the requested stage.
    try:
        if args.stage == "1":
            return _run_stage1(config, args, logger)
        elif args.stage == "2":
            return _run_stage2(config, args, logger)
        elif args.stage == "3":
            return _run_stage3(config, args, logger)
        elif args.stage == "all":
            return _run_all(config, args, logger)
        elif args.stage == "api":
            return _start_api(config, logger)
        else:
            logger.error("Unknown stage: %s", args.stage)
            return 1
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        return 1


def _run_stage1(config, args, logger) -> int:
    """Run Stage 1 for one or more AOIs."""
    from geoai.pipeline.stage1 import run_stage1

    aoi_names = args.aoi or [a.name for a in config.processing.aois]
    if not aoi_names:
        logger.error("No AOIs configured. Check configs/processing.yaml.")
        return 1

    success = True
    for aoi_name in aoi_names:
        logger.info("Stage 1 — AOI: %s", aoi_name)
        try:
            result = run_stage1(
                platform_config=config,
                aoi_name=aoi_name,
                run_id=args.run_id if len(aoi_names) == 1 else None,
            )
            logger.info(
                "Stage 1 complete — %s: change=%.2f ha objects=%d",
                aoi_name,
                result.prediction_summary.get("change_area_ha", 0),
                result.object_summary.get("object_count", 0),
            )
        except Exception as exc:
            logger.error("Stage 1 failed for AOI '%s': %s", aoi_name, exc)
            success = False

    return 0 if success else 1


def _run_stage2(config, args, logger) -> int:
    """Run Stage 2 using a stored Stage 1 result."""
    from geoai.pipeline.stage1 import Stage1Result, run_stage1
    from geoai.pipeline.stage2 import run_stage2

    aoi_names = args.aoi or [a.name for a in config.processing.aois]
    success = True

    for aoi_name in aoi_names:
        try:
            # Re-run Stage 1 to get the result object, or load from disk.
            # For simplicity, Stage 2 re-runs Stage 1 when called standalone.
            logger.info("Stage 1 (prerequisite for Stage 2) — AOI: %s", aoi_name)
            s1 = run_stage1(config, aoi_name=aoi_name, run_id=args.run_id)

            logger.info("Stage 2 — AOI: %s", aoi_name)
            s2 = run_stage2(
                stage1_result=s1,
                platform_config=config,
                compute_shap=args.compute_shap,
            )
            logger.info(
                "Stage 2 complete — %s: %d objects classified.",
                aoi_name, len(s2.objects),
            )
        except Exception as exc:
            logger.error("Stage 2 failed for AOI '%s': %s", aoi_name, exc)
            success = False

    return 0 if success else 1


def _run_stage3(config, args, logger) -> int:
    """Run Stage 3 batch pipeline."""
    from geoai.pipeline.stage3 import run_stage3

    result = run_stage3(
        platform_config=config,
        project_id=args.project_id,
        aoi_names=args.aoi,
        compute_shap=args.compute_shap,
    )
    logger.info(
        "Stage 3 complete — %d AOIs processed, %d failed.",
        len(result.aoi_results), len(result.failed_aois),
    )
    return 0 if not result.failed_aois else 1


def _run_all(config, args, logger) -> int:
    """Run Stage 1 + Stage 2 for specified AOIs."""
    from geoai.pipeline.stage1 import run_stage1
    from geoai.pipeline.stage2 import run_stage2

    aoi_names = args.aoi or [a.name for a in config.processing.aois]
    success = True

    for aoi_name in aoi_names:
        try:
            s1 = run_stage1(config, aoi_name=aoi_name)
            s2 = run_stage2(s1, config, compute_shap=args.compute_shap)
            logger.info(
                "All stages complete — %s: "
                "change=%.2f ha objects=%d classified=%d",
                aoi_name,
                s1.prediction_summary.get("change_area_ha", 0),
                s1.object_summary.get("object_count", 0),
                len(s2.objects),
            )
        except Exception as exc:
            logger.error("Pipeline failed for AOI '%s': %s", aoi_name, exc)
            success = False

    return 0 if success else 1


def _start_api(config, logger) -> int:
    """Start the FastAPI REST API server."""
    try:
        import uvicorn
        from geoai.api.router import create_app

        app = create_app(cors_origins=config.api.cors_origins)
        logger.info(
            "Starting API server — host=%s port=%d",
            config.api.host, config.api.port,
        )
        uvicorn.run(
            app,
            host=config.api.host,
            port=config.api.port,
            log_level="info",
        )
        return 0
    except ImportError:
        logger.error(
            "uvicorn is not installed. Install with: pip install uvicorn"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
