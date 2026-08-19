import os
import json
import streamlit as st
import pandas as pd
from pathlib import Path
from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.experiments.benchmark_runner import BenchmarkRunner

class DatasetExplorerPage:
    """Streamlit dashboard page for pluggable public change detection datasets and benchmarking."""

    def __init__(self) -> None:
        self.runner = BenchmarkRunner()

    def render(self) -> None:
        st.title("📁 Public Dataset Explorer & Benchmarks")
        st.markdown(
            "This interface implements the **Multi-Dataset Benchmarking & Explorer** (Milestone 11). "
            "It validates registry constraints, checks CRS and coordinate integrity, estimates quality scoring metrics, "
            "and runs standardized benchmarking across all registered datasets."
        )

        # Retrieve registry identifiers
        public_ids = ["levir_cd", "oscd", "s2looking"]
        
        # Load or populate datasets
        for p_id in public_ids:
            DatasetRegistry.load_public_dataset(p_id)

        selected_dataset_id = st.sidebar.selectbox("Select Public Dataset to Inspect:", options=public_ids)
        
        # Load registry metadata
        meta = DatasetRegistry.get_dataset_metadata(selected_dataset_id)
        dataset = DatasetRegistry.load_public_dataset(selected_dataset_id)

        if not meta or not dataset:
            st.error(f"Failed to load dataset '{selected_dataset_id}' from registry.")
            return

        # Load Statistics Report if available
        stats_path = Path("outputs/datasets") / f"{selected_dataset_id}_stats.json"
        stats = {}
        if stats_path.exists():
            try:
                with open(stats_path, "r", encoding="utf-8") as f:
                    stats = json.load(f)
            except Exception:
                pass

        # Load Benchmark campaign details
        benchmark_summary_path = Path("outputs/benchmarks/benchmark_summary.json")
        benchmarks = {}
        if benchmark_summary_path.exists():
            try:
                with open(benchmark_summary_path, "r", encoding="utf-8") as f:
                    benchmarks = json.load(f)
            except Exception:
                pass

        # Visual layout structure: Sidebar stats details
        st.sidebar.markdown(f"### 📋 {selected_dataset_id.upper()} Quick Specs")
        st.sidebar.markdown(f"**Resolution**: `{meta.spatial_resolution_m}m` / pixel")
        st.sidebar.markdown(f"**Sensor**: `{meta.sensor_eo}`")
        st.sidebar.markdown(f"**Default CRS**: `{meta.crs}`")
        st.sidebar.markdown(f"**Licensing**: `{meta.licensing}`")
        st.sidebar.markdown(f"**Versioning (Raw/Pre)**: `v{meta.versioning.raw_version}` / `v{meta.versioning.preprocessing_version}`")

        # ----------------------------------------------------
        # LAYOUT TABS
        # ----------------------------------------------------
        tab_summary, tab_matrix, tab_quality, tab_benchmark = st.tabs([
            "🔎 Dataset Overview & Previews",
            "📊 Capability Matrix",
            "🛡️ Quality & Integrity Diagnostics",
            "🏆 Standardized Benchmarking Campaign"
        ])

        # TAB 1: OVERVIEW & PREVIEWS
        with tab_summary:
            col_l, col_r = st.columns([2, 1])
            with col_l:
                st.markdown("### 🗺️ Visual Composites & Spatial Overlay")
                
                # Check for preview image
                preview_img_path = Path("outputs/datasets") / f"{selected_dataset_id}_previews" / "preview_composite.png"
                if not preview_img_path.exists():
                    # Generate preview on the fly if needed
                    from geoai.datasets.statistics_reporter import DatasetStatisticsReporter
                    DatasetStatisticsReporter.generate_report(dataset)
                    
                if preview_img_path.exists():
                    st.image(str(preview_img_path), use_container_width=True)
                else:
                    st.info("No visual preview available. Trigger Quality Diagnostics to generate overlays.")
                    
            with col_r:
                st.markdown("### 📊 Pixel Distributions")
                if stats:
                    st.markdown("**Pixel Frequency Breakdown**")
                    df_dist = pd.DataFrame({
                        "Class": ["No-Change (0)", "Change (1)"],
                        "Ratio": [stats.get("no_change_ratio", 0.95), stats.get("change_ratio", 0.05)]
                    })
                    st.bar_chart(df_dist.set_index("Class"))
                    
                    st.metric("Spatial Autocorrelation (Moran's I)", f"{stats.get('spatial_autocorrelation_moran_i', 0.0):.4f}")
                else:
                    st.info("Run Diagnostics to calculate distribution profiles.")

            st.markdown("---")
            st.markdown("### 📄 BibTeX Citation Reference")
            st.code(meta.bibtex, language="latex")

        # TAB 2: CAPABILITY MATRIX
        with tab_matrix:
            st.markdown("### 🏆 Platform Capability Alignment Matrix")
            st.markdown("This matrix verifies compatibility across downstream model architectures and sensor configurations.")
            
            caps = meta.capabilities.capabilities
            
            # Form grid layout for badges
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.markdown("#### Model Families")
                st.markdown(f"**Classical ML**: {'🟢 Supported' if caps.get('Classical_ML') else '🔴 Unsupported'}")
                st.markdown(f"**CNNs**: {'🟢 Supported' if caps.get('CNN') else '🔴 Unsupported'}")
                st.markdown(f"**Transformers**: {'🟢 Supported' if caps.get('Transformer') else '🔴 Unsupported'}")
                st.markdown(f"**Foundation Models**: {'🟢 Supported' if caps.get('Foundation_Model') else '🔴 Unsupported'}")
            with col_b2:
                st.markdown("#### Radiometric Modalities")
                st.markdown(f"**Optical Imagery**: {'🟢 Supported' if caps.get('Optical') else '🔴 Unsupported'}")
                st.markdown(f"**SAR (Radar)**: {'🟢 Supported' if caps.get('SAR') else '🔴 Unsupported'}")
                st.markdown(f"**Multispectral Bands**: {'🟢 Supported' if caps.get('Multispectral') else '🔴 Unsupported'}")
            with col_b3:
                st.markdown("#### Target Classification")
                st.markdown(f"**Binary Changes**: {'🟢 Supported' if caps.get('Binary') else '🔴 Unsupported'}")
                st.markdown(f"**Multiclass Changes**: {'🟢 Supported' if caps.get('Multiclass') else '🔴 Unsupported'}")

        # TAB 3: QUALITY DIAGNOSTICS
        with tab_quality:
            st.markdown("### 🛡️ Quality & Integrity Diagnostics")
            
            # Check validation files
            from geoai.datasets.dataset_validator import validate_dataset_integrity
            is_valid, errors = validate_dataset_integrity(dataset)
            
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("#### Validation Checks")
                if is_valid:
                    st.success("🟢 Dataset structure is perfectly valid and matches the GeoAI schema.")
                else:
                    st.error("🔴 Integrity validation failures identified:")
                    for err in errors:
                        st.write(f"- {err}")
            with col_v2:
                st.markdown("#### Quality Metrics")
                
                # Trigger Scoring Diagnostics
                if st.button("Run Diagnostic Quality Assessment"):
                    with st.spinner("Scoring dataset alignment, clouds, and boundary complexity..."):
                        from geoai.datasets.quality_scorer import DatasetQualityScorer
                        from geoai.datasets.statistics_reporter import DatasetStatisticsReporter
                        
                        score, details = DatasetQualityScorer.evaluate_quality(dataset)
                        DatasetStatisticsReporter.generate_report(dataset)
                        
                        st.metric("Unified Dataset Quality Score", f"{score:.1f} / 100")
                        
                        # Display sub scores
                        st.markdown(f"- **Spatial Alignment Score**: `{details['alignment_score']:.1f}`")
                        st.markdown(f"- **Cloud-Free Score**: `{details['cloud_free_score']:.1f}`")
                        st.markdown(f"- **Label Consistency (Autocorrelation)**: `{details['label_consistency_score']:.1f}`")
                        st.markdown(f"- **Mean Shift Offset**: `{details['mean_offset_pixels']:.2f} px`")
                        
                        st.rerun()
                else:
                    # Show cached scoring details if available
                    # Default placeholders
                    st.info("Assessment ready to run.")

        # TAB 4: BENCHMARKING
        with tab_benchmark:
            st.markdown("### 🏆 Standardized Benchmarking Campaign")
            st.markdown("Run a standard benchmark suite across all registered datasets using identical experiment hyperparameters.")
            
            col_bm1, col_bm2 = st.columns([1, 2])
            with col_bm1:
                st.markdown("#### Start Benchmark Campaign")
                selected_models = st.multiselect("Select Benchmark Models:", ["rf_baseline_v1", "extra_trees", "xgboost"], default=["rf_baseline_v1"])
                quick_mode = st.checkbox("Quick Mode (Subsample patches)", value=True)
                sample_limit = st.slider("Sample Tiles Limit", 2, 20, 5)
                
                if st.button("🚀 Run Standardized Campaign"):
                    with st.spinner("Executing benchmark loop across public datasets..."):
                        runner = BenchmarkRunner()
                        res = runner.run_campaign(
                            dataset_ids=["levir_cd", "oscd", "s2looking"],
                            model_ids=selected_models,
                            quick_mode=quick_mode,
                            sample_limit=sample_limit
                        )
                        st.success("Campaign benchmark completed!")
                        st.rerun()
                        
            with col_bm2:
                st.markdown("#### Campaign Leaderboard")
                if benchmarks:
                    rows = []
                    for d_id, models in benchmarks.items():
                        for m_id, met in models.items():
                            rows.append({
                                "Dataset": d_id,
                                "Model": m_id,
                                "IoU": met.get("iou", 0.0),
                                "F1 Score": met.get("f1", 0.0),
                                "ECE": met.get("ece", 0.0),
                                "Training Time (s)": met.get("train_time_sec", 0.0),
                                "Inference Time (s)": met.get("inference_time_sec", 0.0),
                                "Dataset Quality": met.get("quality_score", 0.0)
                            })
                    df_bm = pd.DataFrame(rows)
                    st.dataframe(df_bm, use_container_width=True)
                else:
                    st.info("No benchmark campaign history found. Run standard campaigns to display leaderboards.")
