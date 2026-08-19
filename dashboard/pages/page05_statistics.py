import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dashboard.services.config_service import ConfigService
from dashboard.services.outputs_service import OutputService
from dashboard.utils.file_utils import read_text_file


class StatisticsPage:
    def __init__(self) -> None:
        self.config_service = ConfigService()
        self.output_service = OutputService(
            self.config_service.get_export_base_dir().as_posix()
        )

    def render(self) -> None:
        st.title("Statistics")
        st.markdown(
            "View summary statistics from Stage 1 pipeline reports and project comparisons."
        )

        run_ids = self.output_service.list_runs(include_stage2=True)
        if not run_ids:
            st.warning("No completed runs are available.")
            return

        run_id = st.selectbox("Select a run", run_ids)
        reports = self.output_service.list_reports(run_id)
        report_text = None
        if reports:
            report_text = read_text_file(reports[0])

        if report_text:
            st.subheader("Stage 1 Report Summary")
            st.code(report_text[:3000])
        else:
            st.warning("No Stage 1 report available for the selected run.")

        # Load object CSV to generate charts
        csv_path = self.output_service.find_csv(run_id)
        if csv_path and csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    st.divider()
                    st.subheader("Statistical Distributions")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if "class" in df.columns:
                            st.markdown("#### Class Distribution")
                            class_counts = df["class"].value_counts()
                            fig, ax = plt.subplots(figsize=(6, 4))
                            fig.patch.set_facecolor('#0b0f17')
                            ax.set_facecolor('#0d131f')
                            ax.tick_params(colors='#e6edf3')
                            ax.xaxis.label.set_color('#e6edf3')
                            ax.yaxis.label.set_color('#e6edf3')
                            ax.title.set_color('#e6edf3')
                            
                            colors = ['#58a6ff', '#bc8cff', '#3fb950', '#ff7b72', '#d8b4fe', '#79c0ff']
                            class_counts.plot(kind="bar", ax=ax, color=colors[:len(class_counts)])
                            ax.set_title("Object Count by Class")
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.info("Class distribution data not available (requires Stage 2 run).")
                            
                    with col2:
                        if "confidence" in df.columns:
                            st.markdown("#### Confidence Score Distribution")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            fig.patch.set_facecolor('#0b0f17')
                            ax.set_facecolor('#0d131f')
                            ax.tick_params(colors='#e6edf3')
                            ax.xaxis.label.set_color('#e6edf3')
                            ax.yaxis.label.set_color('#e6edf3')
                            ax.title.set_color('#e6edf3')
                            
                            ax.hist(df["confidence"].dropna(), bins=15, color='#8957e5', edgecolor='#1f293d', alpha=0.85)
                            ax.set_title("Confidence Frequency")
                            ax.set_xlabel("Confidence Score")
                            ax.set_ylabel("Count")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.info("Confidence score distribution data not available.")
                            
                    col3, col4 = st.columns(2)
                    with col3:
                        area_col = "area_m2" if "area_m2" in df.columns else ("area_px" if "area_px" in df.columns else None)
                        if area_col:
                            st.markdown(f"#### Object Size Distribution ({'m²' if area_col == 'area_m2' else 'px'})")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            fig.patch.set_facecolor('#0b0f17')
                            ax.set_facecolor('#0d131f')
                            ax.tick_params(colors='#e6edf3')
                            ax.xaxis.label.set_color('#e6edf3')
                            ax.yaxis.label.set_color('#e6edf3')
                            ax.title.set_color('#e6edf3')
                            
                            ax.hist(df[area_col].dropna(), bins=15, color='#3fb950', edgecolor='#1f293d', alpha=0.85)
                            ax.set_title("Object Size Frequency")
                            ax.set_xlabel(f"Area ({'m²' if area_col == 'area_m2' else 'px'})")
                            ax.set_ylabel("Count")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                    with col4:
                        if "ndvi_chg" in df.columns and "sar_chg" in df.columns:
                            st.markdown("#### NDVI Shifts vs SAR Shifts")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            fig.patch.set_facecolor('#0b0f17')
                            ax.set_facecolor('#0d131f')
                            ax.tick_params(colors='#e6edf3')
                            ax.xaxis.label.set_color('#e6edf3')
                            ax.yaxis.label.set_color('#e6edf3')
                            ax.title.set_color('#e6edf3')
                            
                            ax.scatter(df["ndvi_chg"], df["sar_chg"], color='#58a6ff', alpha=0.6, edgecolors='none')
                            ax.set_title("Indices Change Distribution")
                            ax.set_xlabel("NDVI Change")
                            ax.set_ylabel("SAR Change")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
            except Exception as e:
                st.error(f"Failed to generate plots from CSV: {e}")

        comparison = self.output_service.get_comparison_data()
        if comparison:
            st.divider()
            st.subheader("Comparison Summary")
            st.json(comparison)


statistics_page = StatisticsPage()
