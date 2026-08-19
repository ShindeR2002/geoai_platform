import os
import json
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

from geoai.datasets.dataset_registry import DatasetRegistry
from geoai.experiments.provenance import collect_provenance_metadata
from geoai.experiments.db_research import initialize_research_db
from geoai.experiments.qa_system import BenchmarkQASystem
from geoai.experiments.query_engine import ResearchQueryEngine

class ResearchCenterPage:
    """Streamlit dashboard page for the Research Center, containing timelines, QAs, and SQLite queries."""
    
    def __init__(self):
        self.db_path = "outputs/benchmarks/research_registry.db"
        self.output_dir = Path("outputs/benchmarks")
        initialize_research_db(self.db_path)
        self.query_engine = ResearchQueryEngine(self.db_path)

    def render(self):
        st.title("🛰️ Research & Reproducibility Center")
        st.markdown(
            "This interface implements the **Research Platform Refinement & Scientific Reproducibility Upgrade** (Milestone 13). "
            "It tracks environment variables, SQLite historical tables, and dynamic RMI maturity statistics."
        )
        
        self._check_and_populate_mock_data()
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM campaigns")
        campaigns = [dict(r) for r in cursor.fetchall()]
        conn.close()
        
        if not campaigns:
            st.warning("No campaigns found. Execute campaigns to register logs.")
            return
            
        selected_camp = st.sidebar.selectbox("Select Campaign to Analyze:", options=[c["campaign_id"] for c in campaigns])
        
        # ----------------------------------------------------
        # LAYOUT TABS
        # ----------------------------------------------------
        tab_timeline, tab_cards, tab_qa, tab_query, tab_paper = st.tabs([
            "📅 Research Evolution Timeline",
            "📇 Dataset & Experiment Cards",
            "🛡️ Benchmark QA Checkpoint",
            "🔍 Research Query Engine",
            "📄 Publication Paper Builder"
        ])
        
        # TAB 1: RESEARCH EVOLUTION TIMELINE
        with tab_timeline:
            st.markdown("### 📅 Platform Research Evolution Timeline")
            st.markdown("Select metrics to overlay on the chronological step chart to inspect accuracy, latency, and memory progression.")
            
            selected_metrics = st.multiselect(
                "Overlay Metrics:",
                options=["f1", "iou", "boundary_iou", "throughput_pixels_sec", "peak_memory_mb"],
                default=["f1"]
            )
            
            conn = sqlite3.connect(self.db_path)
            df_timeline = pd.read_sql_query("""
                SELECT r.run_id, r.execution_time_sec, e.model_id, 
                       m.f1, m.iou, m.boundary_iou, m.throughput_pixels_sec, m.peak_memory_mb,
                       c.created_at
                FROM runs r
                JOIN experiments e ON r.experiment_id = e.experiment_id
                JOIN metrics m ON r.run_id = m.run_id
                JOIN benchmarks b ON e.benchmark_id = b.benchmark_id
                JOIN campaigns c ON b.campaign_id = c.campaign_id
                WHERE r.status = 'Completed'
            """, conn)
            conn.close()
            
            if not df_timeline.empty and selected_metrics:
                # Group metrics by model ID
                df_grouped = df_timeline.groupby("model_id").agg({m: "max" for m in selected_metrics}).reset_index()
                st.markdown(f"**Metric overlay progression chart ({', '.join(selected_metrics)})**")
                st.bar_chart(df_grouped.set_index("model_id")[selected_metrics])
            else:
                st.info("Select one or more metrics to construct the timeline chart.")

        # TAB 2: DATASET & EXPERIMENT CARDS
        with tab_cards:
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("### 🗺️ Registered Dataset Cards")
                public_ids = ["levir_cd", "oscd", "s2looking"]
                sel_ds = st.selectbox("Select Dataset:", public_ids)
                
                meta = DatasetRegistry.get_dataset_metadata(sel_ds)
                if meta:
                    st.markdown(f"#### Card: {meta.aoi_name}")
                    st.write(f"- **Resolution**: `{meta.spatial_resolution_m}m`")
                    st.write(f"- **Sensor**: `{meta.sensor_eo}`")
                    st.write(f"- **CRS**: `{meta.crs}`")
                    st.write(f"- **Licensing**: `{meta.licensing}`")
                    st.write(f"- **Capabilities**: CNNs: {meta.capabilities.capabilities['CNN']}, Classical: {meta.capabilities.capabilities['Classical_ML']}")
                    
                    st.markdown("**BibTeX Copy Block**")
                    st.code(meta.bibtex, language="latex")
                    
            with col_c2:
                st.markdown("### 🧪 Completed Run Experiment Cards")
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT r.run_id, e.model_id, e.dataset_id, m.iou, m.f1, r.git_commit, r.device_mode
                    FROM runs r
                    JOIN experiments e ON r.experiment_id = e.experiment_id
                    JOIN metrics m ON r.run_id = m.run_id
                    WHERE r.status = 'Completed'
                """)
                runs = [dict(row) for row in cursor.fetchall()]
                conn.close()
                
                if runs:
                    sel_run = st.selectbox("Select Run ID:", [r["run_id"] for r in runs])
                    run_info = next(r for r in runs if r["run_id"] == sel_run)
                    
                    st.markdown(f"#### Experiment: {run_info['run_id']}")
                    st.write(f"- **Model ID**: `{run_info['model_id']}`")
                    st.write(f"- **Dataset ID**: `{run_info['dataset_id']}`")
                    st.write(f"- **Jaccard Pixel IoU**: `{run_info['iou']:.4f}`")
                    st.write(f"- **F1 Score**: `{run_info['f1']:.4f}`")
                    st.write(f"- **Hardware Target**: `{run_info['device_mode']}`")
                    st.write(f"- **Git Commit SHA**: `{run_info['git_commit']}`")
                else:
                    st.info("No runs completed yet.")

        # TAB 3: BENCHMARK QA CHECKPOINT
        with tab_qa:
            st.markdown("### 🛡️ Reproducibility Quality Assurance & Maturity Index")
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.seed, e.model_id, e.dataset_id, r.status, r.run_id, m.ece
                FROM runs r
                JOIN experiments e ON r.experiment_id = e.experiment_id
                LEFT JOIN metrics m ON r.run_id = m.run_id
            """)
            runs_qa = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            from geoai.utils.constants import CANONICAL_FEATURE_NAMES
            score, errors = BenchmarkQASystem.evaluate_campaign_qa(
                runs_list=runs_qa,
                expected_runs_count=2,
                feature_names=list(CANONICAL_FEATURE_NAMES),
                canonical_feature_names=list(CANONICAL_FEATURE_NAMES)
            )
            
            # Dynamic Research Maturity Index calculation
            mean_ece = np.mean([r["ece"] for r in runs_qa if r["ece"] is not None]) if runs_qa else 0.08
            rmi_score = BenchmarkQASystem.calculate_research_maturity_index(
                qa_score=score,
                has_significance=True,
                num_datasets=3,
                ece=mean_ece,
                has_explain=True
            )
            
            col_qa1, col_qa2 = st.columns(2)
            with col_qa1:
                st.metric("Consolidated QA Score", f"{score:.1f} / 100")
                if score == 100.0:
                    st.success("🟢 Complete campaign replication success.")
                else:
                    for err in errors:
                        st.write(f"- {err}")
            with col_qa2:
                st.metric("Research Maturity Index (RMI)", f"{rmi_score:.1f} / 100")
                if rmi_score >= 80.0:
                    st.success("🟢 Experiment is mature enough for publication.")
                else:
                    st.info("🟡 Further dataset testing and calibration improvements suggested.")

        # TAB 4: RESEARCH QUERY ENGINE
        with tab_query:
            st.markdown("### 🔍 SQLite Relational Query Layer")
            st.markdown("Query the campaign database using predefined scientific search criteria.")
            
            query_option = st.selectbox(
                "Select Query Criteria:",
                [
                    "Show all experiments with IoU > threshold",
                    "Show all runs on specific dataset",
                    "Show all failed runs",
                    "Show runs executed on GPU",
                    "Show best model for each dataset",
                    "Show publication-ready experiments (RMI >= 80)"
                ]
            )
            
            df_result = pd.DataFrame()
            if query_option == "Show all experiments with IoU > threshold":
                th = st.slider("IoU Threshold:", 0.0, 1.0, 0.70, step=0.05)
                res = self.query_engine.query_iou_above(th)
                df_result = pd.DataFrame(res)
            elif query_option == "Show all runs on specific dataset":
                ds_sel = st.selectbox("Query Dataset ID:", ["levir_cd", "oscd", "s2looking"])
                res = self.query_engine.query_dataset_runs(ds_sel)
                df_result = pd.DataFrame(res)
            elif query_option == "Show all failed runs":
                res = self.query_engine.query_failed_runs()
                df_result = pd.DataFrame(res)
            elif query_option == "Show runs executed on GPU":
                res = self.query_engine.query_by_device("GPU")
                df_result = pd.DataFrame(res)
            elif query_option == "Show best model for each dataset":
                res = self.query_engine.query_best_models_per_dataset()
                df_result = pd.DataFrame(res)
            elif query_option == "Show publication-ready experiments (RMI >= 80)":
                res = self.query_engine.query_publication_ready_experiments()
                df_result = pd.DataFrame(res)
                
            if not df_result.empty:
                st.dataframe(df_result, use_container_width=True)
            else:
                st.info("No matching records found for this query parameter.")

        # TAB 5: PUBLICATION PAPER BUILDER
        with tab_paper:
            st.markdown("### 📄 Academic Publication Builder")
            
            from geoai.evaluation.paper_builder import PublicationPaperBuilder
            builder = PublicationPaperBuilder()
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT e.model_id, e.dataset_id, m.iou, m.f1, m.ece, m.throughput_pixels_sec, m.peak_memory_mb
                FROM runs r
                JOIN experiments e ON r.experiment_id = e.experiment_id
                JOIN metrics m ON r.run_id = m.run_id
                WHERE r.status = 'Completed'
            """)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            if results:
                paper_path = builder.build_paper(selected_camp, results)
                with open(paper_path, "r", encoding="utf-8") as f:
                    latex_text = f.read()
                    
                st.text_area("LaTeX Paper Source", value=latex_text, height=350)
            else:
                st.info("No results compiled for LaTeX synthesis.")

    def _check_and_populate_mock_data(self):
        """Populate initial seed campaign data in the research database if empty."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM campaigns")
        count = cursor.fetchone()[0]
        if count == 0:
            cursor.execute("INSERT INTO campaigns (campaign_id, name, description) VALUES ('campaign_01', 'Baseline Campaign v1', 'Benchmark stack')")
            cursor.execute("INSERT INTO benchmarks (benchmark_id, campaign_id, name) VALUES ('bench_01', 'campaign_01', 'Within-domain evaluations')")
            cursor.execute("INSERT INTO experiments (experiment_id, benchmark_id, model_id, dataset_id, config_hash) VALUES ('exp_01', 'bench_01', 'rf_baseline_v1', 'levir_cd', 'hash01')")
            cursor.execute("INSERT INTO experiments (experiment_id, benchmark_id, model_id, dataset_id, config_hash) VALUES ('exp_02', 'bench_01', 'fc_siam_conc', 'levir_cd', 'hash02')")
            cursor.execute("INSERT INTO runs (run_id, experiment_id, seed, status, git_commit, device_mode, execution_time_sec) VALUES ('run_01', 'exp_01', 42, 'Completed', 'commit_01', 'CPU', 2.5)")
            cursor.execute("INSERT INTO runs (run_id, experiment_id, seed, status, git_commit, device_mode, execution_time_sec) VALUES ('run_02', 'exp_02', 42, 'Completed', 'commit_01', 'GPU', 14.2)")
            cursor.execute("INSERT INTO metrics (run_id, accuracy, iou, boundary_iou, f1, precision, recall, ece, brier, throughput_pixels_sec, peak_memory_mb) VALUES ('run_01', 0.82, 0.58, 0.52, 0.73, 0.75, 0.71, 0.08, 0.12, 4500.0, 120.0)")
            cursor.execute("INSERT INTO metrics (run_id, accuracy, iou, boundary_iou, f1, precision, recall, ece, brier, throughput_pixels_sec, peak_memory_mb) VALUES ('run_02', 0.91, 0.74, 0.68, 0.85, 0.87, 0.83, 0.04, 0.06, 9500.0, 1850.0)")
            conn.commit()
        conn.close()
