"""
Generalisation pipeline for the GeoAI Platform.

Provides a thin wrapper around the Stage 1 pipeline for running the
generalisation test on the Dholera AOI (and any other AOIs used to test
model generalisation beyond the training site).

This module formalises the Version 1 Notebook 09 end-to-end generalisation
test as a reusable, configuration-driven pipeline call. It is used by the
Stage 3 batch pipeline and by regression tests.

Single responsibility: run Stage 1 inference on a specified generalisation
AOI using the production model.

Position in dependency hierarchy: pipeline (depends on pipeline/stage1).
"""

import logging
from pathlib import Path
from typing import Optional

from geoai.core.config import PlatformConfig
from geoai.pipeline.stage1 import Stage1Result, run_stage1

logger = logging.getLogger(__name__)


def run_generalisation(
    platform_config: PlatformConfig,
    aoi_name: str = "Dholera",
    run_id: Optional[str] = None,
    compute_probabilities: bool = False,
    generate_visualisations: bool = True,
) -> Stage1Result:
    """Run Stage 1 change detection on a generalisation AOI.

    Executes the complete Stage 1 pipeline on the specified AOI using the
    production model (rf_enhanced.pkl). This is the direct platform equivalent
    of Version 1 Notebook 09 (Generalization Test).

    The Dholera AOI is the primary generalisation test case used to validate
    that the model trained on the PS10 urban AOI generalises to a different
    geographic location (Dholera Special Investment Region, Gujarat).

    Args:
        platform_config: Full platform configuration. The AOI must be defined
            in ``processing.aois``.
        aoi_name: Name of the AOI to process. Default ``'Dholera'``.
        run_id: Optional run identifier. Auto-generated if None.
        compute_probabilities: If True, compute prediction probability raster.
        generate_visualisations: If True, generate RGB overlay images.

    Returns:
        :class:`~geoai.pipeline.stage1.Stage1Result` for the generalisation run.

    Raises:
        ConfigurationError: If the AOI is not in the platform configuration.
        StageExecutionError: If the pipeline fails.
    """
    logger.info(
        "Running generalisation test on AOI '%s'.", aoi_name
    )

    result = run_stage1(
        platform_config=platform_config,
        aoi_name=aoi_name,
        run_id=run_id,
        compute_probabilities=compute_probabilities,
        generate_visualisations=generate_visualisations,
        generate_ps10_submission=False,  # Generalisation runs don't produce PS10 packages.
    )

    logger.info(
        "Generalisation test complete — AOI='%s' "
        "change_px=%d objects=%d.",
        aoi_name,
        result.prediction_summary.get("change_pixels", 0),
        result.object_summary.get("object_count", 0),
    )
    return result
