from pathlib import Path
from typing import Dict, Any

def generate_thesis_report(metrics: Dict[str, Any], output_path: Path) -> None:
    """
    Generate publication-ready Markdown and LaTeX summary for research publication,
    including advanced diagnostics, error taxonomy, class-wise object statistics,
    and evidence-based scientific findings.
    """
    landscape = metrics.get("spatial", {}).get("landscape", {})
    patch_size = metrics.get("spatial", {}).get("patch_size_distribution", {})
    runtime = metrics.get("runtime", {}).get("latency_sec", {})
    system = metrics.get("runtime", {}).get("system", {})
    accuracy = metrics.get("bootstrap", {})
    validation = metrics.get("validation", {})
    val_status = "PASSED" if validation.get("passed_all", False) else "DEVIATED / WARNED"
    
    # Class-wise Object Metrics
    class_wise = metrics.get("class_wise_objects", {})
    class_metrics = class_wise.get("class_metrics", {})
    
    # Taxonomy
    taxonomy = metrics.get("errors", {}).get("taxonomy", {})
    total_tax_errors = sum(taxonomy.values())
    
    # Findings
    findings = metrics.get("scientific_findings", [])
    findings_str = "\n".join([f"- **Finding {idx+1}**: {f}" for idx, f in enumerate(findings)])
    
    lines = [
        f"# GeoAI Scientific Research & Benchmarking Platform",
        f"## Thesis Report & Academic Publication Summary",
        f"\n**Validation Status**: `{val_status}`",
        f"\n### 1. Methodology & Experimental Configuration",
        f"This evaluation was executed in accordance with the locked platform pipeline contract. "
        f"Features were dynamically aligned with the model capability specifications to ensure zero count mismatches. "
        f"Hyperparameters: {metrics.get('hyperparameters', {})}",
        f"\n- **Model ID**: `{metrics.get('model_id')}`",
        f"- **Dataset ID**: `{metrics.get('dataset_id')}`",
        f"- **AOI Name**: `{metrics.get('aoi_name')}`",
        f"- **Test Split Ratio**: `{metrics.get('test_size', 0.2) * 100:.1f}% holdout`",
        f"- **Random split seed**: `{metrics.get('random_state', 42)}`",
        f"\n### 2. Statistical Accuracy Metrics",
        f"The following performance metrics were evaluated on the stratified test split. "
        f"Confidence intervals (CIs) were computed using bootstrapping over {metrics.get('bootstrap', {}).get('bootraw_run', 200)} iterations at a 95% confidence level.",
        f"\n| Performance Metric | Calculated Mean | 95% CI Lower Bound | 95% CI Upper Bound |",
        f"| :--- | :--- | :--- | :--- |",
        f"| **F1 Score** | {accuracy.get('f1', {}).get('mean', 0.0):.6f} | {accuracy.get('f1', {}).get('ci_lower', 0.0):.6f} | {accuracy.get('f1', {}).get('ci_upper', 0.0):.6f} |",
        f"| **IoU (Jaccard Index)** | {accuracy.get('iou', {}).get('mean', 0.0):.6f} | {accuracy.get('iou', {}).get('ci_lower', 0.0):.6f} | {accuracy.get('iou', {}).get('ci_upper', 0.0):.6f} |",
        f"\n### 3. Spatial GIS Landscape Statistics",
        f"- **Total Valid Land Evaluated**: {landscape.get('total_valid_area_ha', 0.0):.4f} hectares",
        f"- **Change Objects Extracted**: {landscape.get('n_objects', 0)}",
        f"- **Object Density**: {landscape.get('object_density_per_ha', 0.0):.4f} objects/ha",
        f"- **Nearest Neighbor Centroid Distance**: {landscape.get('mean_nearest_neighbor_meters', 0.0):.2f} meters",
        f"- **Mean Patch Size**: {patch_size.get('mean_size_ha', 0.0):.4f} ha | **Max Patch Size**: {patch_size.get('max_size_ha', 0.0):.4f} ha",
        f"- **Missing GIS Layers**: {metrics.get('spatial', {}).get('skipped_gis_layers', [])}",
        f"\n### 4. Calibration & Diagnostics",
        f"- **Expected Calibration Error (ECE)**: {metrics.get('calibration', {}).get('ece', 0.0):.4f}",
        f"- **Maximum Calibration Error (MCE)**: {metrics.get('calibration', {}).get('mce', 0.0):.4f}",
        f"- **Brier Score**: {metrics.get('calibration', {}).get('brier', 0.0):.4f}",
        f"- **Boundary Error Ratio**: {metrics.get('errors', {}).get('boundary_error_ratio', 0.0):.4f}",
        f"\n### 5. Class-wise Object Metrics Table",
        f"\n| Class Name | TP | FP | FN | Precision | Recall | F1 | IoU |",
        f"| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    for cls, cm in class_metrics.items():
        lines.append(f"| {cls} | {cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['precision']:.4f} | {cm['recall']:.4f} | {cm['f1']:.4f} | {cm['iou']:.4f} |")
        
    lines.extend([
        f"\n### 6. Error Taxonomy Frequency Table",
        f"\n| Error Category | Frequency Count | Frequency Percentage |",
        f"| :--- | :---: | :---: |"
    ])
    
    for cat, val in taxonomy.items():
        pct = (val / total_tax_errors * 100.0) if total_tax_errors > 0 else 0.0
        lines.append(f"| {cat} | {val} | {pct:.2f}% |")
        
    lines.extend([
        f"\n### 7. Scientific Findings & Interpretation",
        findings_str,
        f"\n### 8. Computational Efficiency & Latency profile",
        f"- **Peak RAM Footprint**: {system.get('peak_memory_mb', 0.0):.2f} MB",
        f"- **Estimated Model Size**: {system.get('model_size_mb', 0.0):.4f} MB",
        f"- **Inference Throughput**: {metrics.get('runtime', {}).get('throughput', {}).get('inference_pixels_per_sec', 0.0):,.2f} pixels/sec",
        f"- **Pipeline Latency Breakdown (s)**: Training={runtime.get('training_time', 0.0):.4f}s | Inference={runtime.get('inference_time', 0.0):.4f}s | GIS Objects={runtime.get('object_extraction', 0.0):.4f}s",
        f"\n---\n",
        f"### Appendix: LaTeX Academic Code Blocks",
        f"Copy-paste the LaTeX blocks below directly into your scientific publications or thesis document.",
        f"\n```latex",
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{GeoAI Platform Scientific Benchmarking Results: {metrics.get('model_id')}}}",
        f"\\label{{tab:geoai_results}}",
        f"\\begin{{tabular}}{{lccc}}",
        f"\\hline",
        f"Metric & Mean & 95\\% CI Lower & 95\\% CI Upper \\\\",
        f"\\hline",
        f"F1 Score & {accuracy.get('f1', {}).get('mean', 0.0):.4f} & {accuracy.get('f1', {}).get('ci_lower', 0.0):.4f} & {accuracy.get('f1', {}).get('ci_upper', 0.0):.4f} \\\\",
        f"Jaccard IoU & {accuracy.get('iou', {}).get('mean', 0.0):.4f} & {accuracy.get('iou', {}).get('ci_lower', 0.0):.4f} & {accuracy.get('iou', {}).get('ci_upper', 0.0):.4f} \\\\",
        f"\\hline",
        f"\\end{{tabular}}",
        f"\\end{{table}}",
        f"\n",
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{Class-wise Change Object Diagnostics}}",
        f"\\label{{tab:class_wise_results}}",
        f"\\begin{{tabular}}{{lccccccc}}",
        f"\\hline",
        f"Class Name & TP & FP & FN & Precision & Recall & F1-Score & IoU \\\\",
        f"\\hline"
    ])
    
    for cls, cm in class_metrics.items():
        lines.append(f"{cls.replace('_', ' ')} & {cm['tp']} & {cm['fp']} & {cm['fn']} & {cm['precision']:.4f} & {cm['recall']:.4f} & {cm['f1']:.4f} & {cm['iou']:.4f} \\\\")
        
    lines.extend([
        f"\\hline",
        f"\\end{{tabular}}",
        f"\\end{{table}}",
        f"\n",
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{Spatial Error Taxonomy Distribution}}",
        f"\\label{{tab:error_taxonomy}}",
        f"\\begin{{tabular}}{{lcc}}",
        f"\\hline",
        f"Error Category & Count & Percentage \\\\",
        f"\\hline"
    ])
    
    for cat, val in taxonomy.items():
        pct = (val / total_tax_errors * 100.0) if total_tax_errors > 0 else 0.0
        lines.append(f"{cat} & {val} & {pct:.2f}\\% \\\\")
        
    lines.extend([
        f"\\hline",
        f"\\end{{tabular}}",
        f"\\end{{table}}",
        f"```",
        f"\n---\n",
        f"End of Thesis Report Summary."
    ])
    
    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

def generate_generalization_report(results: dict, output_path: Path) -> None:
    """
    Generate publication-ready Markdown and LaTeX report summarizing cross-AOI generalisation,
    domain shift analysis, similarity score qualitative bins, and PCA/UMAP comparisons.
    """
    import time
    datasets = results["datasets"]
    transfer = results["transfer_results"]
    similarity = results["similarity_matrix"]
    difficulty = results["aoi_difficulty"]
    gap = results["generalization_gap"]
    loao = results["loao"]
    
    lines = [
        "# Cross-AOI Generalization & Domain Robustness Scientific Report",
        "## GeoAI Platform Generalization Benchmark (Milestone 9)",
        f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "\n### 1. Introduction & Methodology",
        "This campaign evaluates model performance degradation under spatial domain shift. "
        "Three evaluation protocols are defined:",
        "- **Protocol A**: Training and testing on the same AOI (within-domain benchmark).",
        "- **Protocol B**: Pairwise cross-AOI transfer (out-of-domain robustness benchmark).",
        "- **Protocol C**: Leave-One-AOI-Out (generalization to completely unseen geography).",
        "\nDomain shift is measured quantitatively via feature-wise Population Stability Index (PSI), "
        "Jensen-Shannon Divergence (JSD), and Correlation cos-similarity.",
        "\n### 2. Dataset Similarity Matrix",
        "The Dataset Similarity Score (0-100) blends feature distributions and correlation matrix alignment.",
        "\n| Source | Target | Similarity Score | Qualitative Category | Mean PSI | Mean JSD |",
        "| :--- | :--- | :---: | :--- | :---: | :---: |"
    ]
    
    for src in datasets:
        for tgt in datasets:
            sim = similarity[src][tgt]
            lines.append(
                f"| {src} | {tgt} | {sim['similarity_score']:.2f} | **{sim['similarity_label']}** | "
                f"{sim['details']['mean_psi']:.4f} | {sim['details']['mean_jsd']:.4f} |"
            )
            
    lines.extend([
        "\n### 3. Protocol B: Pairwise Cross-AOI Performance Transfer Grid",
        "Scores represent Jaccard Pixel IoU on the target's test partition. The diagonal represents Protocol A.",
        "\n| Source \\ Target | " + " | ".join([f"`{d}`" for d in datasets]) + " |",
        "| :--- | " + " | ".join([":---:" for _ in datasets]) + " |"
    ])
    
    for src in datasets:
        row_str = f"| **{src}** | " + " | ".join([f"{transfer[src][tgt]['iou']:.4f}" for tgt in datasets]) + " |"
        lines.append(row_str)
        
    lines.extend([
        "\n### 4. Generalization Gap Matrix",
        "Generalization Gap represents the delta between within-domain baseline performance and transfer performance: "
        "$\\text{IoU}_{\\text{src}} - \\text{IoU}_{\\text{tgt}}$. Higher values indicate severe performance drop under transfer.",
        "\n| Source \\ Target | " + " | ".join([f"`{d}`" for d in datasets]) + " |",
        "| :--- | " + " | ".join([":---:" for _ in datasets]) + " |"
    ])
    
    for src in datasets:
        row_str = f"| **{src}** | " + " | ".join([f"{gap[src][tgt]:.4f}" for tgt in datasets]) + " |"
        lines.append(row_str)
        
    lines.extend([
        "\n### 5. AOI Classification Difficulty Rankings",
        "Difficulty Index is calculated as $1.0 - \\text{mean}(\\text{IoU}_{\\text{target}})$, measuring target region complexity.",
        "\n| AOI Name | Difficulty Index | Average Target IoU |",
        "| :--- | :---: | :---: |"
    ])
    
    for d_id in sorted(datasets, key=lambda x: difficulty[x], reverse=True):
        avg_iou = 1.0 - difficulty[d_id]
        lines.append(f"| {d_id} | {difficulty[d_id]:.4f} | {avg_iou:.4f} |")
        
    lines.extend([
        "\n### 6. Protocol C: Leave-One-AOI-Out Validation",
    ])
    
    if loao["active"]:
        lines.extend([
            "Leave-One-AOI-Out (LOAO) validation represents generalization to an unseen target AOI by joint training on all other AOIs.",
            "\n| Held-Out Target | Leave-One-Out IoU |",
            "| :--- | :---: |"
        ])
        for k, v in loao["results"].items():
            lines.append(f"| {k} | {v['iou']:.4f} |")
    else:
        lines.append(f"\n> ⚠️ **LOAO Status**: Disabled ({loao['message']})")
        
    # Append LaTeX blocks
    lines.extend([
        "\n---\n",
        "### Appendix: LaTeX Academic Code Blocks",
        "Copy-paste the LaTeX blocks below directly into your scientific publications or thesis document.",
        "\n```latex",
        f"\\begin{{table}}[htbp]",
        f"\\centering",
        f"\\caption{{Cross-AOI Pixel-level Transfer Matrix (IoU) and Generalization Gap}}",
        f"\\label{{tab:cross_aoi_results}}",
        f"\\begin{{tabular}}{{l" + "c" * len(datasets) + "}}",
        f"\\hline",
        f"Source \\\\ Target & " + " & ".join([d.replace('_', '\\_') for d in datasets]) + " \\\\",
        f"\\hline"
    ])
    
    for src in datasets:
        row_cells = []
        for tgt in datasets:
            val = transfer[src][tgt]['iou']
            row_cells.append(f"{val:.4f}")
        lines.append(f"{src.replace('_', '\\_')} & " + " & ".join(row_cells) + " \\\\")
        
    lines.extend([
        f"\\hline",
        f"\\end{{tabular}}",
        f"\\end{{table}}",
        f"```",
        f"\n---\n",
        f"End of Cross-AOI Generalization Report Summary."
    ])
    
    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
