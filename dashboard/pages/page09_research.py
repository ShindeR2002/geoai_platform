import time
from pathlib import Path
import streamlit as st
import pandas as pd
import json

from geoai.experiments.experiment_manager import ExperimentManager
from geoai.experiments.experiment_runner import ExperimentRunner
from geoai.experiments.experiment_registry import get_experiment_model
from geoai.experiments.leaderboard import Leaderboard
from geoai.experiments.ablation import AblationManager
from geoai.experiments.benchmark import BenchmarkSuite
from geoai.experiments.benchmark_report import generate_benchmark_markdown_report

class ResearchPage:
    """Streamlit dashboard page for Research & Experiments."""
    
    def __init__(self) -> None:
        self.manager = ExperimentManager()
        self.runner = ExperimentRunner()
        self.ablation_manager = AblationManager()
        self.leaderboard = Leaderboard()

    def render(self) -> None:
        st.title("🛰️ Research & Experiments Framework")
        st.markdown(
            "This interface implements the **GeoAI Experiment & Benchmark Framework** (Milestone 2). "
            "It provides a pluggable dataset registry, strict model capability validation, isolated "
            "artifact tracking, locked baseline checks, and automated ablation studies."
        )
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Leaderboard & Benchmarking",
            "🚀 Launch Experiment",
            "🔍 Run Explorer",
            "🧪 Ablation Study Board",
            "🛰️ Preprocessing & Feature Campaigns"
        ])
        
        # TAB 1: LEADERBOARD & BENCHMARKING
        with tab1:
            self._render_leaderboard_tab()
            
        # TAB 2: LAUNCH EXPERIMENT
        with tab2:
            self._render_launch_tab()
            
        # TAB 3: RUN EXPLORER
        with tab3:
            self._render_explorer_tab()
            
        # TAB 4: ABLATION BOARD
        with tab4:
            self._render_ablation_tab()

        # TAB 5: PREPROCESSING CAMPAIGN BOARD
        with tab5:
            self._render_preprocessing_campaign_tab()

    def _render_leaderboard_tab(self) -> None:
        st.subheader("Leaderboard")
        st.markdown("All completed experiments are ranked by **IoU (Jaccard Index)** descending, adhering to the locked baseline split conditions.")
        
        entries = self.leaderboard.load()
        if not entries:
            st.info("No leaderboard entries recorded yet. Run an experiment config to generate entries.")
            return
            
        df = pd.DataFrame(entries)
        # Rename column names for friendly display
        df_display = df.rename(columns={
            "experiment_id": "Experiment ID",
            "model_id": "Model ID",
            "aoi_name": "AOI",
            "dataset_id": "Dataset",
            "f1": "F1 Score",
            "iou": "IoU (Jaccard)",
            "precision": "Precision",
            "recall": "Recall",
            "runtime_sec": "Runtime (s)",
            "memory_mb": "RAM (MB)",
            "date": "Date Run"
        })
        
        st.dataframe(df_display, use_container_width=True)
        
        # Display Provenance & Citations Details for Models
        st.markdown("### 📚 Model Publication & Provenance Registry")
        model_selection = st.selectbox(
            "Select model to view scientific paper details, venue, DOI, and implementation code:",
            options=["rf_enhanced_v1", "rf_baseline_v1", "xgboost", "catboost", "lightgbm", "siamese", "changeformer"]
        )
        
        try:
            model_wrapper = get_experiment_model(model_selection)
            capabilities = model_wrapper.get_capabilities()
            
            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                st.markdown(f"**Model Type**: `{capabilities.model_type}`")
                st.markdown(f"**Status**: `{capabilities.training_status}`")
                st.markdown(f"**Required Packages**: `{capabilities.requirements}`")
            with col_meta2:
                st.markdown(f"**Expected Inputs**: `{capabilities.expected_input_channels}`")
                st.markdown(f"**Implemented**: `{'🟢 Yes' if model_wrapper.is_implemented() else '🔴 No (Integrity Locked)'}`")
                
            # Provenance attributes if unimplemented model
            if hasattr(model_wrapper, "provenance"):
                prov = model_wrapper.provenance
                st.info(
                    f"**Paper Title**: *{prov.paper_title}*\n\n"
                    f"**Authors**: {prov.authors} ({prov.publication_year})\n\n"
                    f"**Venue**: {prov.venue} | **DOI**: [{prov.doi}](https://doi.org/{prov.doi})\n\n"
                    f"**Repository**: [{prov.repository}]({prov.repository})\n\n"
                    f"**License**: `{prov.license}`"
                )
                with st.expander("Show BibTeX Citation"):
                    st.code(prov.citation, language="bibtex")
            else:
                st.success("This model is fully implemented in the GeoAI core package and ready for production training and inference.")
        except Exception as e:
            st.error(f"Error reading model properties: {e}")
            
        # Running Benchmark Suite
        st.markdown("---")
        st.markdown("### 📊 Generate Benchmark Comparison Report")
        st.markdown("Contrast all completed candidate experiments against the locked reference baseline (`exp_rf_baseline`).")
        if st.button("Generate Benchmark Suite Report"):
            try:
                suite = BenchmarkSuite()
                candidate_ids = [e["experiment_id"] for e in entries if e["experiment_id"] != "exp_rf_baseline"]
                bench_results = suite.run_benchmark(candidate_ids)
                
                report_md = generate_benchmark_markdown_report(bench_results)
                st.markdown(report_md)
                
                # Write to disk as benchmark report
                report_path = Path("outputs/experiments/benchmark_report.md")
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(report_md, encoding="utf-8")
                st.success(f"Benchmark report saved to `{report_path}`")
            except Exception as e:
                st.error(f"Failed to generate benchmark: {e}")

    def _render_launch_tab(self) -> None:
        st.subheader("Launch Experiment Run")
        st.markdown("Select a pre-registered experiment configuration YAML from `configs/experiments/` to execute.")
        
        configs = self.manager.list_configs()
        if not configs:
            st.warning("No experiment config files found under `configs/experiments/`.")
            return
            
        config_names = [c.name for c in configs]
        selected_config_name = st.selectbox("Choose Configuration:", options=config_names)
        selected_config_path = configs[config_names.index(selected_config_name)]
        
        # Load and show YAML preview
        try:
            with open(selected_config_path, "r", encoding="utf-8") as f:
                config_content = f.read()
            with st.expander("View YAML Config Content"):
                st.code(config_content, language="yaml")
        except Exception as e:
            st.error(f"Error reading config: {e}")
            return
            
        # Button to run
        if st.button("🚀 Execute Experiment Run"):
            log_placeholder = st.empty()
            status_placeholder = st.empty()
            
            status_placeholder.info(f"Initiating run for {selected_config_name}...")
            
            try:
                # Read experiment ID to locate log file
                import yaml
                parsed = yaml.safe_load(config_content)
                exp_id = parsed.get("experiment_id", "run_temp")
                log_file = Path("outputs/experiments") / exp_id / "logs" / "execution.log"
                
                # Run experiment runner in-process
                # (Can stream log file lines during execution)
                with st.spinner("Executing training & evaluation pipeline..."):
                    experiment = self.runner.run(selected_config_path)
                    
                if experiment.get_state() == "Completed":
                    status_placeholder.success(f"Experiment completed successfully! State: {experiment.get_state()}")
                    st.balloons()
                else:
                    status_placeholder.error(f"Experiment execution finished with state: {experiment.get_state()}")
                    
                # Load and display logs
                if log_file.exists():
                    with open(log_file, "r", encoding="utf-8") as f:
                        log_lines = f.read()
                    with st.expander("View Execution Logs"):
                        st.code(log_lines)
            except Exception as e:
                status_placeholder.error(f"Execution Error: {e}")
                
                # Load logs if available anyway
                try:
                    exp_id = yaml.safe_load(config_content).get("experiment_id", "")
                    log_file = Path("outputs/experiments") / exp_id / "logs" / "execution.log"
                    if log_file.exists():
                        with open(log_file, "r", encoding="utf-8") as f:
                            log_lines = f.read()
                        with st.expander("View Error Execution Logs"):
                            st.code(log_lines)
                except Exception:
                    pass

    def _render_explorer_tab(self) -> None:
        st.subheader("Run Explorer")
        st.markdown("Browse details, visual metrics curves, baseline compliance logs, and export outputs of completed runs.")
        
        experiments = self.manager.list_experiments(include_archived=True)
        if not experiments:
            st.info("No completed experiment runs found in outputs directory.")
            return
            
        exp_ids = [e["experiment_id"] for e in experiments]
        selected_exp_id = st.selectbox("Select Experiment Run:", options=exp_ids)
        
        # Load experiment metadata
        exp = self.manager.get_experiment(selected_exp_id)
        if not exp:
            st.error("Failed to load experiment metadata.")
            return
            
        # Display Info & Metrics
        st.markdown(f"### Run details: `{selected_exp_id}`")
        col_det1, col_det2 = st.columns(2)
        with col_det1:
            st.markdown(f"**Model ID**: `{exp.get('model_id')}`")
            st.markdown(f"**Dataset ID**: `{exp.get('dataset_id')}`")
            st.markdown(f"**AOI Name**: `{exp.get('aoi_name')}`")
            st.markdown(f"**State**: `{exp.get('state')}`")
        with col_det2:
            st.markdown(f"**Timestamp**: `{exp.get('timestamp')}`")
            st.markdown(f"**Split Random Seed**: `{exp.get('random_seed')}`")
            
        # Baseline lock deviations
        deviations = exp.get("deviations", [])
        if deviations:
            st.warning(f"⚠️ **Baseline Lock Deviations Identified**:\n\n" + "\n".join([f"- {d}" for d in deviations]))
        else:
            st.success("🟢 baseline Compliance Locked (Fully matches pipeline contract specifications)")
            
        st.markdown("---")
        st.markdown("### 📈 Evaluation Metrics & Execution Profile")
        metrics = exp.get("metrics", {})
        if metrics:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("IoU (Jaccard Index)", f"{metrics.get('iou', 0.0):.6f}")
            col_m2.metric("F1 Score", f"{metrics.get('f1', 0.0):.6f}")
            col_m3.metric("Precision", f"{metrics.get('precision', 0.0):.6f}")
            col_m4.metric("Recall", f"{metrics.get('recall', 0.0):.6f}")
            
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric("Training Time", f"{metrics.get('train_time_sec', 0.0):.4f} s")
            col_p2.metric("Model Size", f"{metrics.get('model_size_mb', 0.0):.4f} MB")
            col_p3.metric("Memory RAM Peak", f"{metrics.get('memory_usage_mb', 0.0):.2f} MB")
        else:
            st.info("Metrics not calculated for this experiment.")
            
        # Plot visual metrics
        st.markdown("---")
        st.markdown("### 📊 Metrics Plots")
        run_dir = Path("outputs/experiments") / selected_exp_id
        
        plot_col1, plot_col2, plot_col3 = st.columns(3)
        with plot_col1:
            roc_path = run_dir / "plots" / "roc_curve.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC Curve", use_container_width=True)
            else:
                st.info("No ROC plot available")
        with plot_col2:
            pr_path = run_dir / "plots" / "pr_curve.png"
            if pr_path.exists():
                st.image(str(pr_path), caption="PR Curve", use_container_width=True)
            else:
                st.info("No PR plot available")
        with plot_col3:
            conf_path = run_dir / "plots" / "confusion.png"
            if conf_path.exists():
                st.image(str(conf_path), caption="Confusion Matrix", use_container_width=True)
            else:
                st.info("No Confusion matrix plot available")
                
        # Outputs and downloads
        st.markdown("---")
        st.markdown("### 📁 Serialized Artifact Outputs")
        st.markdown(f"All artifacts are isolated in `{run_dir}`:")
        
        tif_path = run_dir / "prediction.tif"
        geojson_path = run_dir / "objects.geojson"
        shp_path = run_dir / "objects.shp"
        pkl_path = run_dir / "model.pkl"
        csv_path = run_dir / "statistics.csv"
        
        st.markdown(f"- **Trained Model Weights**: `{pkl_path if pkl_path.exists() else 'Missing'}`")
        st.markdown(f"- **Spatial Prediction Mask**: `{tif_path if tif_path.exists() else 'Missing'}`")
        st.markdown(f"- **Significant Change Objects GeoJSON**: `{geojson_path if geojson_path.exists() else 'Missing'}`")
        st.markdown(f"- **Object Boundaries Shapefiles**: `{shp_path if shp_path.exists() else 'Missing'}`")
        st.markdown(f"- **Object Tabular Statistics CSV**: `{csv_path if csv_path.exists() else 'Missing'}`")
        
        # Option to Archive / Delete this run
        st.markdown("---")
        st.markdown("### ⚙️ Lifecycle Administration")
        current_state = exp.get("state", "Draft")
        if current_state != "Archived":
            if st.button("📁 Archive this Experiment Run"):
                if self.manager.update_experiment_state(selected_exp_id, "Archived"):
                    st.success(f"Run {selected_exp_id} state successfully updated to Archived!")
                    st.rerun()
                else:
                    st.error("Failed to update lifecycle state.")
        else:
            if st.button("🟢 Unarchive this Experiment Run"):
                if self.manager.update_experiment_state(selected_exp_id, "Completed"):
                    st.success(f"Run {selected_exp_id} state restored to Completed!")
                    st.rerun()
                else:
                    st.error("Failed to restore lifecycle state.")

    def _render_ablation_tab(self) -> None:
        st.subheader("🧪 Ablation Study Board")
        st.markdown(
            "An ablation study evaluates how individual spectral bands, SAR backscatter, or "
            "calculated indices impact change detection accuracy, Precision, Recall, and F1/IoU."
        )
        
        st.markdown("### 1. Setup Ablation Experiment")
        target_experiment = st.selectbox(
            "Select Target Baseline Run to Ablate:",
            options=["exp_rf_baseline", "exp_rf_enhanced"]
        )
        
        # Check if target experiment has run
        target_meta = Path("outputs/experiments") / target_experiment / "metadata.json"
        if not target_meta.exists():
            st.warning(f"Target run `{target_experiment}` must be executed under 'Launch Experiment' before you can run ablation study on it.")
            
        features_to_remove = st.multiselect(
            "Select features to remove (zero out) during study:",
            options=["NDVI", "NDBI", "NDWI", "SAR", "Red", "Green", "Blue"]
        )
        
        ablation_run_suffix = st.text_input("Ablation Study Run ID Suffix:", value="drop_ndvi")
        
        if st.button("🧪 Execute Ablation Study Run"):
            if not features_to_remove:
                st.error("Please select at least one feature to remove.")
                return
            if not ablation_run_suffix:
                st.error("Please specify a suffix for the ablation run ID.")
                return
                
            status_placeholder = st.empty()
            status_placeholder.info("Running ablation execution... zeroing out features...")
            
            try:
                with st.spinner("Zeroing features & retraining model weights..."):
                    ablation_exp = self.ablation_manager.run_ablation(
                        target_experiment_id=target_experiment,
                        features_to_remove=features_to_remove,
                        ablation_id=ablation_run_suffix
                    )
                
                if ablation_exp.get_state() == "Completed":
                    status_placeholder.success("Ablation run completed!")
                    
                    # Contrast against locked baseline
                    compare_results = self.ablation_manager.compare_to_target(
                        target_experiment_id=target_experiment,
                        ablation_id=ablation_run_suffix
                    )
                    
                    if compare_results:
                        st.markdown("### 📊 Performance Contrast Card")
                        
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            st.markdown(f"**Baseline Model**: `{compare_results.get('exp1_model')}`")
                            st.markdown(f"**Ablated Model**: `{compare_results.get('exp2_model')}`")
                            
                        # Show Delta Table
                        deltas = compare_results.get("metric_deltas", {})
                        contrast_rows = []
                        for m_name, d_info in deltas.items():
                            contrast_rows.append({
                                "Metric": m_name,
                                "Baseline Value": d_info.get("val1"),
                                "Ablated Value": d_info.get("val2"),
                                "Delta": d_info.get("delta")
                            })
                            
                        st.table(pd.DataFrame(contrast_rows))
                        
                        # Highlighting IoU impact
                        iou_delta = deltas.get("iou", {}).get("delta", 0.0)
                        if iou_delta < 0:
                            st.error(f"Removing `{features_to_remove}` caused IoU to **decrease by {abs(iou_delta):.6f}**. This indicates these features contribute significantly to spatial precision.")
                        elif iou_delta > 0:
                            st.success(f"Removing `{features_to_remove}` caused IoU to **increase by {iou_delta:.6f}**. This may indicate these features introduce noise for this AOI.")
                        else:
                            st.info(f"Removing `{features_to_remove}` had **no impact** on the IoU Jaccard Index.")
                            
                    st.balloons()
                else:
                    status_placeholder.error("Ablation run failed. Check logs.")
            except Exception as e:
                status_placeholder.error(f"Failed executing ablation: {e}")

    def _render_preprocessing_campaign_tab(self) -> None:
        st.subheader("🛰️ Preprocessing & Feature Engineering Campaigns Board")
        st.markdown(
            "This board presents the results of the preprocessing and feature engineering benchmarking campaigns "
            "designed to answer specific research questions about model inputs."
        )
        
        # Check if campaign results exist
        results_path = Path("outputs/preprocessing_benchmark/raw_campaign_results.json")
        if not results_path.exists():
            st.info(
                "Campaign results have not been generated yet. "
                "Please run `python scripts/run_preprocessing_campaign.py` on your terminal to launch the benchmark suite."
            )
            return
            
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception as e:
            st.error(f"Failed to load campaign results: {e}")
            return
            
        # Parse into separate lists
        campaign_a = []
        campaign_b_prod = []
        campaign_b_exp = []
        
        for exp_id, res in results.items():
            if res.get("status") != "Completed":
                continue
            metrics = res.get("metrics", {})
            # Determine campaign and track based on name conventions
            is_preproc = "preproc_" in exp_id
            is_prod = "feat_prod_" in exp_id
            
            item = {
                "Experiment ID": exp_id,
                "IoU (Jaccard)": metrics.get("iou", 0.0),
                "F1 Score": metrics.get("f1", 0.0),
                "Precision": metrics.get("precision", 0.0),
                "Recall": metrics.get("recall", 0.0),
                "Train Time (s)": metrics.get("train_time_sec", 0.0),
                "Inference Latency (s)": metrics.get("inference_time_sec", 0.0),
                "RAM (MB)": metrics.get("memory_usage_mb", 0.0)
            }
            
            if is_preproc:
                campaign_a.append(item)
            elif is_prod:
                campaign_b_prod.append(item)
            else:
                campaign_b_exp.append(item)
                
        # Sort by IoU
        campaign_a = sorted(campaign_a, key=lambda x: x["IoU (Jaccard)"], reverse=True)
        campaign_b_prod = sorted(campaign_b_prod, key=lambda x: x["IoU (Jaccard)"], reverse=True)
        campaign_b_exp = sorted(campaign_b_exp, key=lambda x: x["IoU (Jaccard)"], reverse=True)
        
        # Selector
        campaign_selection = st.radio(
            "Select Campaign to View:",
            options=["Campaign A: Preprocessing Benchmark", "Campaign B: Feature Engineering Benchmark"]
        )
        
        if campaign_selection == "Campaign A: Preprocessing Benchmark":
            st.markdown("### 🏆 Preprocessing Quality Leaderboard")
            st.markdown(
                "**Research Question**: *Does improving the quality of existing input measurements "
                "(without changing the model or expanding the feature space) improve Sentinel-1/2 change detection performance?*"
            )
            if not campaign_a:
                st.info("No Campaign A runs found.")
            else:
                df_a = pd.DataFrame(campaign_a)
                st.dataframe(df_a, use_container_width=True)
                
                # Show horizontal bar chart of IoUs
                st.markdown("#### Jaccard IoU Comparison Chart")
                df_chart = df_a.set_index("Experiment ID")["IoU (Jaccard)"]
                st.bar_chart(df_chart)
                
                # Show resource charts
                st.markdown("#### Computational Footprint Comparison")
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.markdown("**Inference Latency (s)**")
                    st.bar_chart(df_a.set_index("Experiment ID")["Inference Latency (s)"])
                with col_res2:
                    st.markdown("**Peak RAM Footprint (MB)**")
                    st.bar_chart(df_a.set_index("Experiment ID")["RAM (MB)"])
                    
        else: # Campaign B
            st.markdown("### 🏆 Feature Engineering Leaderboard")
            st.markdown(
                "**Research Question**: *Do additional derived spatial, spectral, or texture variables "
                "improve model performance beyond the current feature set?*"
            )
            
            sub_tab1, sub_tab2 = st.tabs([
                "🟢 Production Track (Physical Bands)",
                "⚠️ Experimental Track (NIR Reconstruction Approximations)"
            ])
            
            with sub_tab1:
                st.markdown("#### Production Features (Physical Bands Only)")
                st.markdown(
                    "These experiments utilize only physically grounded, native measurements on disk. "
                    "No reconstructed bands are permitted. **Recommended for final deployment.**"
                )
                if not campaign_b_prod:
                    st.info("No production feature runs found.")
                else:
                    df_prod = pd.DataFrame(campaign_b_prod)
                    st.dataframe(df_prod, use_container_width=True)
                    
                    # Contribution plot
                    st.markdown("#### Performance Contribution (Delta IoU Relative to Baseline)")
                    base_iou = next((x["IoU (Jaccard)"] for x in campaign_b_prod if x["Experiment ID"] == "feat_prod_baseline"), 0.0)
                    df_contrib = df_prod.copy()
                    df_contrib["Delta IoU"] = df_contrib["IoU (Jaccard)"] - base_iou
                    st.bar_chart(df_contrib.set_index("Experiment ID")["Delta IoU"])
                    
            with sub_tab2:
                st.warning(
                    "⚠️ **Experimental Approximation Warning**\n\n"
                    "The configurations below use an algebraically reconstructed Near-Infrared (NIR) band. "
                    "Because physical Band 8 is missing from the raw imagery catalog, SAVI and MSAVI calculations "
                    "are mathematically estimated approximations. These results are for research exploration only "
                    "and must NOT be deployed in production or compared directly against physical baselines."
                )
                if not campaign_b_exp:
                    st.info("No experimental feature runs found.")
                else:
                    df_exp = pd.DataFrame(campaign_b_exp)
                    st.dataframe(df_exp, use_container_width=True)
                    
                    # Contribution plot relative to baseline
                    st.markdown("#### Performance Contribution (Delta IoU Relative to Production Baseline)")
                    base_iou = next((x["IoU (Jaccard)"] for x in campaign_b_prod if x["Experiment ID"] == "feat_prod_baseline"), 0.0)
                    if base_iou == 0.0 and campaign_b_exp:
                        base_iou = campaign_b_exp[-1]["IoU (Jaccard)"]
                    df_exp_contrib = df_exp.copy()
                    df_exp_contrib["Delta IoU"] = df_exp_contrib["IoU (Jaccard)"] - base_iou
                    st.bar_chart(df_exp_contrib.set_index("Experiment ID")["Delta IoU"])
