import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import joblib
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

def compute_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """Calculate the Population Stability Index (PSI) between source (expected) and target (actual)."""
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(expected, percentiles)
    
    # Adjust boundaries slightly to prevent extreme out-of-bounds exclusion
    bins[0] -= 1e-5
    bins[-1] += 1e-5
    
    # Ensure bins are monotonically increasing
    for i in range(1, len(bins)):
        if bins[i] <= bins[i-1]:
            bins[i] = bins[i-1] + 1e-5
            
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Replace zeros to prevent division by zero or log of zero
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)
    
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)

def compute_jsd(expected: np.ndarray, actual: np.ndarray, num_bins: int = 20) -> float:
    """Compute the Jensen-Shannon Divergence (JSD) between source and target feature arrays."""
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    expected_pct, _ = np.histogram(expected, bins=bins)
    actual_pct, _ = np.histogram(actual, bins=bins)
    
    expected_pct = expected_pct / len(expected)
    actual_pct = actual_pct / len(actual)
    
    # JS distance is bounded between 0 and 1
    js_dist = jensenshannon(expected_pct, actual_pct)
    if np.isnan(js_dist):
        return 0.0
    return float(js_dist)

class GeneralizationEvaluator:
    """Scientific evaluation coordinator implementing Protocols A, B, and C, and measuring domain shift."""
    
    def __init__(self, configs_dir: str = "configs", outputs_dir: str = "outputs/generalization") -> None:
        self.configs_dir = Path(configs_dir)
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
    def compute_dataset_similarity(self, X_src: np.ndarray, X_tgt: np.ndarray, feature_names: List[str]) -> Tuple[float, str, Dict[str, Any]]:
        """
        Compute a normalized Dataset Similarity Score (0-100) blending PSI, JSD, and correlation shifts.
        Returns score, qualitative label, and detail metrics.
        """
        num_features = X_src.shape[1]
        
        # 1. Feature-wise PSI & JSD
        psis = []
        jsds = []
        feature_details = {}
        
        for idx, feat in enumerate(feature_names):
            psi_val = compute_psi(X_src[:, idx], X_tgt[:, idx])
            jsd_val = compute_jsd(X_src[:, idx], X_tgt[:, idx])
            psis.append(psi_val)
            jsds.append(jsd_val)
            feature_details[feat] = {"psi": psi_val, "jsd": jsd_val}
            
        mean_psi = np.mean(psis)
        mean_jsd = np.mean(jsds)
        
        # Normalize PSI: clip at 2.0 (representing extreme drift) and map to [0, 1]
        norm_psi = np.clip(mean_psi / 2.0, 0.0, 1.0)
        
        # JSD is already in [0, 1]
        norm_jsd = np.clip(mean_jsd, 0.0, 1.0)
        
        # 2. Correlation dependency shift
        df_src = pd.DataFrame(X_src, columns=feature_names)
        df_tgt = pd.DataFrame(X_tgt, columns=feature_names)
        
        corr_src = df_src.corr().fillna(0.0).values
        corr_tgt = df_tgt.corr().fillna(0.0).values
        
        # Vectorized cosine similarity between correlation matrices
        src_flat = corr_src.ravel()
        tgt_flat = corr_tgt.ravel()
        
        dot_product = np.dot(src_flat, tgt_flat)
        norm_src = np.linalg.norm(src_flat)
        norm_tgt = np.linalg.norm(tgt_flat)
        
        if norm_src > 0 and norm_tgt > 0:
            corr_cos_sim = dot_product / (norm_src * norm_tgt)
        else:
            corr_cos_sim = 1.0
            
        norm_corr_diff = 1.0 - np.clip(corr_cos_sim, 0.0, 1.0)
        
        # Blend the three normalized distance components into a similarity score
        mean_dist = np.mean([norm_psi, norm_jsd, norm_corr_diff])
        similarity_score = float(100.0 * (1.0 - mean_dist))
        similarity_score = max(0.0, min(100.0, similarity_score))
        
        # Classify qualitatively
        if similarity_score >= 90.0:
            label = "Very Similar"
        elif similarity_score >= 75.0:
            label = "Similar"
        elif similarity_score >= 50.0:
            label = "Moderate"
        elif similarity_score >= 25.0:
            label = "Different"
        else:
            label = "Highly Different"
            
        details = {
            "mean_psi": float(mean_psi),
            "mean_jsd": float(mean_jsd),
            "correlation_cosine_similarity": float(corr_cos_sim),
            "norm_psi": float(norm_psi),
            "norm_jsd": float(norm_jsd),
            "norm_corr_diff": float(norm_corr_diff),
            "feature_details": feature_details
        }
        
        return similarity_score, label, details

    def generate_domain_shift_projections(
        self,
        X_src: np.ndarray,
        X_tgt: np.ndarray,
        src_name: str,
        tgt_name: str
    ) -> Dict[str, Any]:
        """Fit PCA and UMAP (or t-SNE fallback) over concatenated source and target samples."""
        max_samples = 2000
        
        # Sample datasets to avoid large visualization runtime
        np.random.seed(42)
        n_src_samples = min(len(X_src), max_samples)
        n_tgt_samples = min(len(X_tgt), max_samples)
        
        idx_src = np.random.choice(len(X_src), n_src_samples, replace=False)
        idx_tgt = np.random.choice(len(X_tgt), n_tgt_samples, replace=False)
        
        X_src_sampled = X_src[idx_src]
        X_tgt_sampled = X_tgt[idx_tgt]
        
        X_combined = np.concatenate([X_src_sampled, X_tgt_sampled], axis=0)
        labels = np.array([0] * n_src_samples + [1] * n_tgt_samples)
        
        # 1. PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_combined)
        
        # 2. UMAP with t-SNE fallback
        has_umap = False
        method_name = "t-SNE"
        
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            X_umap = reducer.fit_transform(X_combined)
            has_umap = True
            method_name = "UMAP"
        except ImportError:
            logger.warning("umap-learn not installed. Falling back to t-SNE for projection.")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            X_umap = reducer.fit_transform(X_combined)
            
        # Plot and save
        plots_path = self.outputs_dir / "plots"
        plots_path.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # PCA Scatter
        ax1.scatter(X_pca[labels == 0, 0], X_pca[labels == 0, 1], c='#4F46E5', label=src_name, alpha=0.6, s=15)
        ax1.scatter(X_pca[labels == 1, 0], X_pca[labels == 1, 1], c='#EF4444', label=tgt_name, alpha=0.6, s=15)
        ax1.set_title("Global Variance Overlay (PCA)")
        ax1.set_xlabel("PCA Component 1")
        ax1.set_ylabel("PCA Component 2")
        ax1.legend()
        
        # UMAP/t-SNE Scatter
        ax2.scatter(X_umap[labels == 0, 0], X_umap[labels == 0, 1], c='#4F46E5', label=src_name, alpha=0.6, s=15)
        ax2.scatter(X_umap[labels == 1, 0], X_umap[labels == 1, 1], c='#EF4444', label=tgt_name, alpha=0.6, s=15)
        ax2.set_title(f"Local Neighborhood Structure Overlay ({method_name})")
        ax2.set_xlabel(f"{method_name} Component 1")
        ax2.set_ylabel(f"{method_name} Component 2")
        ax2.legend()
        
        plot_file = plots_path / f"domain_shift_{src_name.lower()}_{tgt_name.lower()}.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150)
        plt.close()
        
        return {
            "projection_method": method_name,
            "pca_coordinates_src": X_pca[labels == 0].tolist(),
            "pca_coordinates_tgt": X_pca[labels == 1].tolist(),
            "reducer_coordinates_src": X_umap[labels == 0].tolist(),
            "reducer_coordinates_tgt": X_umap[labels == 1].tolist(),
            "plot_path": str(plot_file)
        }
