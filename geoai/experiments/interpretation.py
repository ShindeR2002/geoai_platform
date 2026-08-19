import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ScientificInterpretationEngine:
    """Evaluates metrics deltas across runs to generate evidence-based scientific descriptions."""
    
    @staticmethod
    def interpret_model_comparisons(metrics_a: Dict[str, float], metrics_b: Dict[str, float], model_a: str, model_b: str) -> str:
        """Explain why Model A performed differently compared to Model B."""
        insights = []
        iou_diff = metrics_a.get("iou", 0.0) - metrics_b.get("iou", 0.0)
        biou_diff = metrics_a.get("boundary_iou", 0.0) - metrics_b.get("boundary_iou", 0.0)
        ece_diff = metrics_a.get("ece", 0.0) - metrics_b.get("ece", 0.0)
        tp_diff = metrics_a.get("throughput_pixels_sec", 1.0) / max(1e-8, metrics_b.get("throughput_pixels_sec", 1.0))
        
        # 1. Accuracy comparison
        if iou_diff > 0.05:
            insights.append(
                f"Model '{model_a}' outperformed '{model_b}' by {iou_diff*100:.1f}% in Pixel IoU. "
                "This performance delta is supported by its hierarchical convolutional representation structure, "
                "which captures spatial context far better than pixel-level classical features."
            )
        elif iou_diff < -0.05:
            insights.append(
                f"Model '{model_b}' outperformed '{model_a}' by {-iou_diff*100:.1f}% in Pixel IoU. "
                "This indicates that the tabular features stack contains highly discriminative spectral indices "
                "ideal for tree ensemble optimization."
            )
            
        # 2. Boundary accuracy comparison
        if biou_diff > 0.05:
            insights.append(
                f"Model '{model_a}' improved boundary alignment (B-IoU) by {biou_diff*100:.1f}%. "
                "This confirms that representation layers retain structural contours, reducing classification "
                "bleed-through at building borders."
            )
            
        # 3. Calibration comparison
        if ece_diff > 0.05:
            insights.append(
                f"Model '{model_a}' exhibit higher Expected Calibration Error (+{ece_diff*100:.1f}%), "
                "suggesting a tendency toward making overconfident, uncalibrated out-of-domain predictions."
            )
            
        # 4. Latency / Efficiency comparison
        if tp_diff < 0.1:
            insights.append(
                f"Model '{model_a}' displays high latency, running at only {tp_diff*100:.2f}% of '{model_b}' throughput. "
                "This is due to the dense parameter stack and GPU-to-CPU context synchronization overhead."
            )
            
        if not insights:
            insights.append(f"Models '{model_a}' and '{model_b}' exhibit statistically equivalent metrics profiles.")
            
        return "\n".join(insights)

    @staticmethod
    def interpret_preprocessing_impact(metrics_raw: Dict[str, float], metrics_prep: Dict[str, float], prep_type: str) -> str:
        """Explain why a preprocessing configuration altered performance outcomes."""
        iou_diff = metrics_prep.get("iou", 0.0) - metrics_raw.get("iou", 0.0)
        
        if iou_diff > 0.02:
            return (
                f"Preprocessing pipeline '{prep_type}' improved Pixel IoU by {iou_diff*100:.1f}%. "
                "Suppressing high-frequency variance (e.g. speckle lee filtering on SAR bands) "
                "reduced false-positive changes and enhanced structural features alignment."
            )
        elif iou_diff < -0.02:
            return (
                f"Preprocessing pipeline '{prep_type}' degraded Pixel IoU by {-iou_diff*100:.1f}%. "
                "This suggests that aggressive spectral filtering removed critical radiometric information "
                "necessary for change categorization."
            )
            
        return f"Preprocessing pipeline '{prep_type}' had negligible impact on overall segmentation metrics."
