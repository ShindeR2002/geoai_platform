# Engineering Walkthrough Report — TinyCD Baseline Execution Recovery

> [!NOTE]
> **Reporting Standard Version:** v1.0
> **Repository Commit:** `46f91b1353`
> **Evaluation Timestamp:** 2026-07-10T14:25:28Z
> **Python Version:** 3.12.10
> **Operating System:** Windows

---

## 1. Experimental Setup & Configurations

We successfully recovered the TinyCD Baseline execution profile and finalized the run:
* **Entry Point**: [run_tinycd_quick_profile.py](file:///c:/Users/Rohit/OneDrive/Desktop/Experiqs/geoai_platform/scripts/run_tinycd_quick_profile.py)
* **Configuration**: [tinycd_quick_profile.yaml](file:///c:/Users/Rohit/OneDrive/Desktop/Experiqs/geoai_platform/configs/experiments/tinycd_quick_profile.yaml)
* **Dataset**: `ps10_sentinel_v1` (subsampled to 100 coordinates for quick profile validation)
* **Resume Status**: Option A (Resumed training from Epoch 2, successfully skipping redundant training cycles and executing prediction/evaluation).

---

## 2. Verification Outcomes

### A. Execution & Training Metrics

| Metric | Value |
| --- | --- |
| Accuracy | `0.6299` |
| Precision | `0.0511` |
| Recall | `0.3991` |
| F1 Score | `0.0906` |
| IoU (Jaccard) | `0.0474` |
| Dice | `0.0906` |
| ROC AUC | `0.5045` |
| Average Precision | `0.0500` |
| Balanced Accuracy | `0.5201` |
| ECE | `0.0513` |
| Brier Score | `0.1191` |

### B. Confusion Matrix

| Actual / Predicted | Predicted No-Change | Predicted Change |
| --- | --- | --- |
| **Actual No-Change** | `56,714` (TN) | `31,755` (FP) |
| **Actual Change** | `2,573` (FN) | `1,709` (TP) |

### C. Runtime Profiling

* **Fitting Duration**: `5.1973` seconds
* **Inference Latency**: `173.7842` seconds
* **Peak RAM**: `584.50` MB
* **Model Weight Size**: `54.36` KB (`55,669` bytes)
* **GPU Availability**: CPU Execution Mode
