import numpy as np
import logging
from typing import Tuple, Dict
from geoai.datasets.dataset_schema import UnifiedChangeDetectionDataset

logger = logging.getLogger(__name__)

class DatasetQualityScorer:
    """Evaluates spatial alignment, cloud cover ratios, and label noise to compute a unified Quality Score (0-100)."""
    
    @staticmethod
    def _phase_correlation_offset(img1: np.ndarray, img2: np.ndarray) -> float:
        """Estimate pixel-level registration offset between two images using phase correlation."""
        try:
            # Check shape
            if img1.shape != img2.shape:
                return 10.0 # High penalty for size mismatch
                
            f1 = np.fft.fft2(img1)
            f2 = np.fft.fft2(img2)
            num = f1 * np.conj(f2)
            den = np.abs(num)
            den[den == 0] = 1.0
            r = np.fft.ifft2(num / den)
            peak = np.unravel_index(np.argmax(np.abs(r)), r.shape)
            
            # Convert peak to coordinates relative to center
            dy = peak[0] if peak[0] < r.shape[0] / 2 else peak[0] - r.shape[0]
            dx = peak[1] if peak[1] < r.shape[1] / 2 else peak[1] - r.shape[1]
            return float(np.sqrt(dx**2 + dy**2))
        except Exception:
            return 5.0 # Fallback penalty if calculation fails

    @staticmethod
    def evaluate_quality(dataset: UnifiedChangeDetectionDataset, sample_limit: int = 10) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate dataset quality by sampling tiles and running checks.
        Returns:
            composite_score (float): Score from 0 to 100.
            details (dict): Indvidual diagnostic metrics.
        """
        n_samples = min(len(dataset), sample_limit)
        if n_samples == 0:
            return 0.0, {"alignment_score": 0.0, "cloud_free_score": 0.0, "label_consistency_score": 0.0}
            
        offsets = []
        cloud_ratios = []
        label_noise_scores = []
        
        for idx in range(n_samples):
            try:
                t1, t2, label = dataset.get_pair(idx)
                
                # 1. Alignment check (use first channel for correlation)
                offset = DatasetQualityScorer._phase_correlation_offset(t1[0], t2[0])
                offsets.append(offset)
                
                # 2. Cloud estimation: high-reflectance threshold (RGB > 0.85)
                # Assume optical if channel size >= 3
                if t1.shape[0] >= 3:
                    cloud_mask_t1 = (t1[0] > 0.85) & (t1[1] > 0.85) & (t1[2] > 0.85)
                    cloud_mask_t2 = (t2[0] > 0.85) & (t2[1] > 0.85) & (t2[2] > 0.85)
                    cloud_ratio = (cloud_mask_t1.sum() + cloud_mask_t2.sum()) / (t1[0].size * 2)
                    cloud_ratios.append(cloud_ratio)
                else:
                    cloud_ratios.append(0.0) # Assume SAR/multispectral cloud-free for simplicity
                    
                # 3. Label boundary noise: ratio of boundary pixels to total area
                change_pixels = np.sum(label == 1)
                if change_pixels > 0:
                    # Quick boundary check using shift comparisons
                    shifted_h = np.pad(label, ((1, 0), (0, 0)), mode='edge')[:-1, :]
                    shifted_w = np.pad(label, ((0, 0), (1, 0)), mode='edge')[:, :-1]
                    boundary_pixels = np.sum((label != shifted_h) | (label != shifted_w))
                    # Extremely high boundary/area ratio suggests high fragmentation (noise)
                    noise_ratio = boundary_pixels / change_pixels
                    label_noise_scores.append(noise_ratio)
                else:
                    label_noise_scores.append(0.0)
            except Exception as e:
                logger.warning(f"Error scoring sample {idx}: {e}")
                
        # Aggregate scores
        mean_offset = np.mean(offsets) if offsets else 0.0
        mean_cloud = np.mean(cloud_ratios) if cloud_ratios else 0.0
        mean_noise = np.mean(label_noise_scores) if label_noise_scores else 0.0
        
        # Define component scores
        alignment_score = max(0.0, 100.0 - (mean_offset * 15.0))
        cloud_free_score = max(0.0, 100.0 - (mean_cloud * 100.0))
        label_consistency_score = max(0.0, 100.0 - (mean_noise * 10.0))
        
        composite_score = float(0.4 * alignment_score + 0.3 * cloud_free_score + 0.3 * label_consistency_score)
        
        return composite_score, {
            "alignment_score": float(alignment_score),
            "cloud_free_score": float(cloud_free_score),
            "label_consistency_score": float(label_consistency_score),
            "mean_offset_pixels": float(mean_offset),
            "mean_cloud_cover_ratio": float(mean_cloud),
            "mean_label_fragmentation_ratio": float(mean_noise)
        }
