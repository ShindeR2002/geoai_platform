import numpy as np
import torch
import logging
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

def run_explainability_audit(
    model_wrapper: Any,
    x1: torch.Tensor,
    x2: torch.Tensor,
    visualizations_dir: str
) -> Dict[str, Any]:
    """
    Perform structural and mathematical audits on explainability hooks of DL models.
    """
    import os
    os.makedirs(visualizations_dir, exist_ok=True)
    
    results = {}
    failures = []
    
    # 1. Run twice to check determinism
    model_wrapper.model.eval()
    
    # Run 1
    attn1 = model_wrapper.get_attention_maps(x1, x2)
    tokens1 = model_wrapper.get_token_embeddings(x1, x2)
    
    # Run 2
    attn2 = model_wrapper.get_attention_maps(x1, x2)
    tokens2 = model_wrapper.get_token_embeddings(x1, x2)
    
    # Assert determinism
    det_attn = torch.allclose(attn1, attn2, atol=1e-6)
    det_tokens = torch.allclose(tokens1, tokens2, atol=1e-6)
    
    results["deterministic_outputs"] = {
        "status": "PASS" if (det_attn and det_tokens) else "FAIL",
        "attn_deterministic": det_attn,
        "tokens_deterministic": det_tokens
    }
    if not (det_attn and det_tokens):
        failures.append("Attention maps or token embeddings are not deterministic under identical inputs.")
        
    # 2. Assert value range bounds of attention maps [0, 1]
    attn_min = float(attn1.min().item())
    attn_max = float(attn1.max().item())
    range_valid = (attn_min >= 0.0 and attn_max <= 1.0)
    results["attention_value_range"] = {
        "status": "PASS" if range_valid else "FAIL",
        "min": attn_min,
        "max": attn_max
    }
    if not range_valid:
        failures.append(f"Attention coefficients exceed range bounds [0, 1]: min={attn_min}, max={attn_max}")

    # 3. NaN / Inf checks
    has_nan_attn = torch.isnan(attn1).any().item() or torch.isinf(attn1).any().item()
    has_nan_tok = torch.isnan(tokens1).any().item() or torch.isinf(tokens1).any().item()
    results["nan_inf_detection"] = {
        "status": "PASS" if not (has_nan_attn or has_nan_tok) else "FAIL",
        "nan_in_attention": has_nan_attn,
        "nan_in_tokens": has_nan_tok
    }
    if has_nan_attn or has_nan_tok:
        failures.append("Attention maps or token embeddings contain NaN or infinite values.")

    # 4. Dimension Verification
    # Expected shapes: attn is (B, 1, H, W), tokens is (B, C, H, W)
    B, C_in, H, W = x1.shape
    expected_attn_shape = (B, 1, H, W)
    results["tensor_dimensions"] = {
        "status": "PASS" if (attn1.shape == expected_attn_shape and tokens1.ndim == 4) else "FAIL",
        "attn_shape": list(attn1.shape),
        "tokens_shape": list(tokens1.shape)
    }
    if attn1.shape != expected_attn_shape or tokens1.ndim != 4:
        failures.append(f"Mismatched tensor dimensions: attn={attn1.shape}, tokens={tokens1.shape}")

    # 5. Dimensionality reduction (PCA / UMAP) & Visualization
    try:
        # Convert tokens to numpy for visualization
        tokens_np = tokens1.cpu().numpy() # shape (B, C_out, H, W)
        B_out, C_out, H_out, W_out = tokens_np.shape
        
        # Flatten spatial dimensions to run PCA over channels: (B*H*W, C_out)
        flat_tokens = tokens_np.transpose(0, 2, 3, 1).reshape(-1, C_out)
        
        # Fallback to PCA since UMAP is not standard in python environments
        reducer_name = "PCA"
        try:
            import umap
            reducer = umap.UMAP(n_components=3, random_state=42)
            reduced = reducer.fit_transform(flat_tokens)
            reducer_name = "UMAP"
        except Exception:
            reducer = PCA(n_components=3, random_state=42)
            reduced = reducer.fit_transform(flat_tokens)
            
        # Reshape back to image grid: (B, H, W, 3)
        reduced_grid = reduced.reshape(B_out, H_out, W_out, 3)
        # Normalize to [0, 1] for RGB visualization
        reduced_grid = (reduced_grid - reduced_grid.min()) / (reduced_grid.max() - reduced_grid.min() + 1e-8)
        
        # Save visualization figures
        for idx in range(B_out):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            # Attention Map
            attn_img = attn1[idx, 0].cpu().numpy()
            im0 = axes[0].imshow(attn_img, cmap='hot', interpolation='nearest')
            axes[0].set_title(f"Attention Map (Sample {idx})")
            fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Reduced Tokens
            axes[1].imshow(reduced_grid[idx])
            axes[1].set_title(f"{reducer_name} Tokens Proj (Sample {idx})")
            
            out_img_path = os.path.join(visualizations_dir, f"{model_wrapper.model_id}_explainability_sample_{idx}.png")
            plt.savefig(out_img_path, bbox_inches='tight', dpi=150)
            plt.close()
            
        results["visualization"] = {
            "status": "PASS",
            "reducer_used": reducer_name,
            "saved_visualizations": B_out
        }
    except Exception as e:
        results["visualization"] = {
            "status": "FAIL",
            "error": str(e)
        }
        failures.append(f"Visualization generation failed: {e}")

    results["audit_status"] = "FAIL" if len(failures) > 0 else "PASS"
    results["failures"] = failures
    
    return results

def generate_explainability_report(audit_results: Dict[str, Any], filepath: str) -> None:
    """Generate Markdown report for explainability validation."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filepath, "w") as f:
        f.write("# GeoAI Platform — Explainability Validation Report\n\n")
        f.write(f"- **Execution Timestamp**: {timestamp}\n")
        f.write(f"- **Overall Audit Status**: {audit_results.get('audit_status', 'UNKNOWN')}\n\n")
        
        f.write("## Technical Integrity Checks\n\n")
        f.write("| Verification Area | Status | Audit Findings |\n")
        f.write("| :--- | :--- | :--- |\n")
        
        for k, v in audit_results.items():
            if k in ("audit_status", "failures"):
                continue
            status = v.get("status", "UNKNOWN")
            details = ", ".join([f"{key}={val}" for key, val in v.items() if key != "status"])
            f.write(f"| {k.replace('_', ' ').title()} | {status} | {details} |\n")
            
        f.write("\n## Explainability Visualizations\n\n")
        vis_count = audit_results.get("visualization", {}).get("saved_visualizations", 0)
        if vis_count > 0:
            f.write("Below are the dimensional embeddings and attention map checks:\n\n")
            # Embed image in markdown using relative path from artifacts
            f.write(f"![Explainability Projection Map](example_predictions/changeformer_explainability_sample_0.png)\n")
        else:
            f.write("> [!WARNING]\n")
            f.write("> No explainability maps were saved or generated.\n")
            
        f.write("\n## Integrity Log\n")
        failures = audit_results.get("failures", [])
        if failures:
            f.write("> [!CAUTION]\n")
            for fail in failures:
                f.write(f"> - {fail}\n")
        else:
            f.write("> [!NOTE]\n")
            f.write("> Scientific audit checks: explainability tensor outputs are fully deterministic and mathematically bounded.\n")
