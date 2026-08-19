import os
import platform
from pathlib import Path
import streamlit as st

from dashboard.services.config_service import ConfigService
from dashboard.services.outputs_service import OutputService
from dashboard.utils.file_utils import read_text_file
from dashboard.utils.styles import render_metric_card


class HomePage:
    def __init__(self) -> None:
        self.config_service = ConfigService()
        self.output_service = OutputService(
            self.config_service.get_export_base_dir().as_posix()
        )

    def render(self) -> None:
        st.title("GeoAI Platform Dashboard")
        st.markdown(
            "The dashboard provides quick access to pipeline runs, exports, "
            "spatial artefacts, and model metadata for the GeoAI Platform."
        )

        aoi_names = self.config_service.get_aoi_names()
        default_aoi = self.config_service.get_default_aoi()
        run_ids = self.output_service.list_runs(include_stage2=False)

        col1, col2, col3 = st.columns(3)
        with col1:
            render_metric_card("Configured AOIs", str(len(aoi_names)), "Ready for inference")
        with col2:
            render_metric_card("Stage 1 runs", str(len(run_ids)), "Completed output runs")
        with col3:
            render_metric_card("Export root", self.config_service.get_export_base_dir().name, "outputs/")

        if run_ids:
            latest_run = run_ids[0]
            run_dir = self.output_service.get_run_dir(latest_run)
            report_files = self.output_service.list_reports(latest_run)
            summary_text = "No reports available."
            if report_files:
                summary_text = read_text_file(report_files[0]) or "No report content."

            st.markdown(f"### Latest Run: `{latest_run}`")
            st.markdown(f"*Output directory:* `{run_dir}`")
            with st.expander("Latest run report preview"):
                st.code(summary_text[:2800])

        if default_aoi:
            st.markdown(f"### Default AOI: **{default_aoi}**")

        st.markdown("---")
        st.markdown("### System Status")
        col_sys1, col_sys2, col_sys3, col_sys4 = st.columns(4)
        
        # Check model file presence
        platform_config = self.config_service.get_platform_config()
        model_config = platform_config.model
        model_path = Path(model_config.model_path)
        model_exists = model_path.exists()
        model_status = "🟢 Ready" if model_exists else "🔴 Missing"
        model_detail = f"ID: {model_config.model_id}" if model_exists else f"Expected: {model_config.model_path}"
        
        # Check configs directory
        configs_dir = Path("configs")
        configs_status = "🟢 Configured" if configs_dir.exists() else "🔴 Missing"
        yaml_count = len(list(configs_dir.glob("*.yaml")))
        configs_detail = f"{yaml_count} config files loaded"
        
        # Check OS environment
        sys_os = platform.system()
        os_release = platform.release()
        os_status = f"🟢 {sys_os}"
        os_detail = f"Release: {os_release} | {os.cpu_count() or 1} CPU Cores"
        
        # Check Outputs root
        outputs_dir = self.output_service.output_base_dir
        outputs_exists = outputs_dir.exists()
        outputs_status = "🟢 Active" if outputs_exists else "🟡 Idle"
        outputs_detail = f"{len(list(outputs_dir.iterdir())) if outputs_exists else 0} runs stored"
        
        with col_sys1:
            st.info(f"**Model Registry**\n\n{model_status}\n\n`{model_detail}`")
        with col_sys2:
            st.info(f"**Configuration**\n\n{configs_status}\n\n`{configs_detail}`")
        with col_sys3:
            st.info(f"**Platform OS**\n\n{os_status}\n\n`{os_detail}`")
        with col_sys4:
            st.info(f"**Outputs Directory**\n\n{outputs_status}\n\n`{outputs_detail}`")

        st.markdown("---")
        st.markdown("### Quick Links")
        st.markdown(
            "- Use **Run Pipeline** to execute Stage 1 or Stage 3 batch processing.\n"
            "- Use **Map Viewer** to inspect GeoJSON object boundaries and change overlays.\n"
            "- Use **Exports** to download PS10 submission and object exports.\n"
        )


home_page = HomePage()
