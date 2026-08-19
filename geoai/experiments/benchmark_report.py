from pathlib import Path
from typing import Dict, Any

def generate_benchmark_markdown_report(benchmark_results: Dict[str, Any]) -> str:
    """Generate a clean Markdown summary report comparing candidate runs to the baseline."""
    baseline_id = benchmark_results.get("baseline_id", "exp_rf_baseline")
    baseline_data = benchmark_results.get("baseline_data")
    comparisons = benchmark_results.get("comparisons", {})
    
    lines = [
        f"# GeoAI Research Platform - Scientific Benchmark Summary",
        f"\n**Baseline Experiment Reference**: `{baseline_id}`",
    ]
    
    if not baseline_data:
        lines.append("\n> [!WARNING]")
        lines.append(f"> Baseline run data for `{baseline_id}` was not found. Benchmark metrics cannot be computed relative to the baseline.")
        return "\n".join(lines)
        
    b_metrics = baseline_data.get("metrics", {})
    lines.extend([
        f"- **Baseline Model**: `{baseline_data.get('model_id')}`",
        f"- **Baseline Dataset**: `{baseline_data.get('dataset_id')}`",
        f"- **Baseline IoU (Jaccard)**: {b_metrics.get('iou', 0.0):.6f}",
        f"- **Baseline F1 Score**: {b_metrics.get('f1', 0.0):.6f}",
        f"- **Baseline Train Time**: {b_metrics.get('train_time_sec', 0.0):.4f}s",
        f"\n## Comparative Leaderboard Table",
        "\n| Experiment ID | Model ID | Dataset ID | IoU (Delta) | F1 (Delta) | Runtime Speedup | Memory Ratio | Split/Seed Match |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])
    
    # Add baseline row first
    lines.append(
        f"| `{baseline_id}` (Locked Baseline) | `{baseline_data.get('model_id')}` | `{baseline_data.get('dataset_id')}` | {b_metrics.get('iou', 0.0):.6f} (ref) | {b_metrics.get('f1', 0.0):.6f} (ref) | 1.00x | 1.00x | Yes |"
    )
    
    for cid, comp in comparisons.items():
        m_deltas = comp.get("metric_deltas", {})
        r_comp = comp.get("resource_comparison", {})
        
        iou_info = m_deltas.get("iou", {})
        f1_info = m_deltas.get("f1", {})
        
        iou_val = iou_info.get("val2", 0.0)
        iou_delta = iou_info.get("delta", 0.0)
        iou_str = f"{iou_val:.6f} ({'+' if iou_delta >= 0 else ''}{iou_delta:.6f})"
        
        f1_val = f1_info.get("val2", 0.0)
        f1_delta = f1_info.get("delta", 0.0)
        f1_str = f"{f1_val:.6f} ({'+' if f1_delta >= 0 else ''}{f1_delta:.6f})"
        
        # Runtime speedup: val1 (baseline) / val2 (candidate). If candidate is faster, speedup > 1.0
        train_time_comp = r_comp.get("train_time_sec", {})
        val1_time = train_time_comp.get("val1", 0.0)
        val2_time = train_time_comp.get("val2", 0.0)
        speedup = val1_time / val2_time if val2_time > 0 else 0.0
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        
        # Memory ratio: val2 (candidate) / val1 (baseline). If candidate uses more, ratio > 1.0
        mem_comp = r_comp.get("memory_usage_mb", {})
        val1_mem = mem_comp.get("val1", 0.0)
        val2_mem = mem_comp.get("val2", 0.0)
        mem_ratio = val2_mem / val1_mem if val1_mem > 0 else 0.0
        mem_ratio_str = f"{mem_ratio:.2f}x" if mem_ratio > 0 else "N/A"
        
        match_str = "Yes" if comp.get("dataset_match") and comp.get("seed_match") else "No (Deviations!)"
        
        lines.append(
            f"| `{cid}` | `{comp.get('exp2_model')}` | `{baseline_data.get('dataset_id')}` | {iou_str} | {f1_str} | {speedup_str} | {mem_ratio_str} | {match_str} |"
        )
        
    lines.extend([
        f"\n## Key Scientific Insights",
        "\n- Models are ranked on the global leaderboard by **IoU (Jaccard Index)**, which serves as the primary metric of scientific correctness.",
        "- Any deviation from baseline preprocessing or splitting (Split/Seed Match is 'No') should be scrutinized to ensure fair evaluation comparisons."
    ])
    
    return "\n".join(lines)


def save_benchmark_report(benchmark_results: Dict[str, Any], output_path: Path) -> None:
    """Save the benchmark report to the designated path."""
    md_content = generate_benchmark_markdown_report(benchmark_results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md_content, encoding="utf-8")
