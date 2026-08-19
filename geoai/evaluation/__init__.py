from geoai.evaluation.evaluation_manager import EvaluationManager
from geoai.evaluation.baseline_validator import validate_v1_v2_consistency
from geoai.evaluation.feature_analysis import (
    BaseFeatureImportance,
    RandomForestMDIImportance,
    PermutationFeatureImportance,
    compute_correlations,
    compute_feature_statistics,
)
from geoai.evaluation.error_analysis import analyze_prediction_errors
from geoai.evaluation.spatial_analysis import analyze_spatial_landscape
from geoai.evaluation.runtime_analysis import profile_execution_performance
from geoai.evaluation.statistical_analysis import compute_bootstrap_confidence_intervals
from geoai.evaluation.metrics_report import export_evaluation_metrics
from geoai.evaluation.publication_report import generate_thesis_report
from geoai.evaluation.ablation_manager import AblationStudyManager

__all__ = [
    "EvaluationManager",
    "validate_v1_v2_consistency",
    "BaseFeatureImportance",
    "RandomForestMDIImportance",
    "PermutationFeatureImportance",
    "compute_correlations",
    "compute_feature_statistics",
    "analyze_prediction_errors",
    "analyze_spatial_landscape",
    "profile_execution_performance",
    "compute_bootstrap_confidence_intervals",
    "export_evaluation_metrics",
    "generate_thesis_report",
    "AblationStudyManager",
]
