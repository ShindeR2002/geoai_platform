import json
import csv
from pathlib import Path
from geoai.experiments.experiment import Experiment

def generate_reports(experiment: Experiment, output_dir: Path) -> None:
    """Generate Markdown, JSON, and CSV reports inside the experiment output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON report
    json_path = output_dir / "metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(experiment.to_dict(), f, indent=2, default=str)

    # 2. CSV metrics
    csv_path = output_dir / "metrics.csv"
    metrics = experiment.metrics
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            if isinstance(v, list):
                v = json.dumps(v)
            writer.writerow([k, v])

    # 3. Markdown report
    md_path = output_dir / "report.md"
    content = _build_markdown_report(experiment)
    md_path.write_text(content, encoding="utf-8")


def _build_markdown_report(experiment: Experiment) -> str:
    m = experiment.metrics
    meta = experiment.execution_metadata
    sep = "=" * 60

    cm = m.get("confusion_matrix", [[0, 0], [0, 0]])
    tn = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
    fp = cm[0][1] if len(cm) > 0 and len(cm[0]) > 1 else 0
    fn = cm[1][0] if len(cm) > 1 and len(cm[1]) > 0 else 0
    tp = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0

    deviations_str = "\n".join([f"- {d}" for d in experiment.deviations]) if experiment.deviations else "None (Baseline Compliance Locked)"

    lines = [
        f"# GeoAI Experiment Report: {experiment.experiment_id}",
        f"\n**Timestamp**: {experiment.timestamp}",
        f"**Model ID**: `{experiment.model_id}`",
        f"**Dataset ID**: `{experiment.dataset_id}`",
        f"**AOI Name**: `{experiment.aoi_name}`",
        f"**Lifecycle State**: `{experiment.get_state()}`",
        f"\n## Performance Metrics",
        f"- **Accuracy**: {m.get('accuracy', 0.0):.6f}",
        f"- **Precision**: {m.get('precision', 0.0):.6f}",
        f"- **Recall**: {m.get('recall', 0.0):.6f}",
        f"- **F1 Score**: {m.get('f1', 0.0):.6f}",
        f"- **IoU (Jaccard Index)**: {m.get('iou', 0.0):.6f} *(primary comparison metric)*",
        f"- **Dice Coefficient**: {m.get('dice', 0.0):.6f}",
        f"- **ROC AUC**: {m.get('roc_auc', 0.0):.6f}",
        f"- **Average Precision (AP)**: {m.get('average_precision', 0.0):.6f}",
        f"\n## Resource Profiling",
        f"- **Training Time**: {m.get('train_time_sec', 0.0):.4f} seconds",
        f"- **Inference Latency**: {m.get('inference_time_sec', 0.0):.4f} seconds",
        f"- **Inference Throughput**: {m.get('prediction_throughput', 0.0):,.1f} samples/second",
        f"- **Model Size**: {m.get('model_size_mb', 0.0):.4f} MB",
        f"- **Memory Usage**: {m.get('memory_usage_mb', 0.0):.2f} MB",
        f"\n## Confusion Matrix",
        "```",
        f"               Predicted No-Change    Predicted Change",
        f"True No-Change      {tn:>10,}            {fp:>6,}",
        f"True Change         {fn:>10,}            {tp:>6,}",
        "```",
        f"\n## Baseline Deviations",
        deviations_str,
        f"\n## Environmental & Hardware Metadata",
        f"- **OS**: {meta.get('os')} {meta.get('os_release')}",
        f"- **Python Version**: {meta.get('python_version')}",
        f"- **CPU Cores**: {meta.get('cpu_cores')}",
        f"- **RAM Total**: {meta.get('ram_gb')} GB",
        f"- **GPU**: {meta.get('gpu')}",
        f"- **Git Commit**: `{meta.get('git_commit')}`",
        f"\n{sep}",
        "End of Report",
        sep
    ]
    return "\n".join(lines)
