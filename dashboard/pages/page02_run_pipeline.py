import queue
import logging
import threading
import time
from pathlib import Path
import streamlit as st

from dashboard.services.config_service import ConfigService
from dashboard.services.pipeline_service import PipelineService
from dashboard.services.outputs_service import OutputService
from dashboard.utils.file_utils import human_readable_size


class QueueLogHandler(logging.Handler):
    """Custom logging handler to send logs to a thread-safe Queue."""
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%H:%M:%S"
        ))

    def emit(self, record):
        self.log_queue.put(self.format(record))


def run_in_thread(target_func, args, kwargs, output_dict):
    """Helper to run a pipeline service method inside a thread."""
    try:
        output_dict["result"] = target_func(*args, **kwargs)
        output_dict["status"] = "success"
    except Exception as e:
        output_dict["error"] = e
        output_dict["status"] = "failed"


class RunPipelinePage:
    def __init__(self) -> None:
        self.config_service = ConfigService()
        self.pipeline_service = PipelineService()
        self.output_service = OutputService(
            self.config_service.get_export_base_dir().as_posix()
        )

    def render(self) -> None:
        st.title("Run Pipeline")
        st.markdown(
            "Execute Stage 1 for a single AOI or run a Stage 3 batch project "
            "across configured AOIs directly from the dashboard."
        )

        aoi_names = self.config_service.get_aoi_names()
        default_aoi = self.config_service.get_default_aoi()
        selected_aoi = st.selectbox("Select AOI for Stage 1", aoi_names, index=aoi_names.index(default_aoi) if default_aoi in aoi_names else 0)

        st.markdown("**Stage 1 options**")
        compute_probabilities = st.checkbox("Compute prediction probabilities", value=False)
        generate_visualisations = st.checkbox("Generate visualisations", value=True)
        generate_ps10_submission = st.checkbox("Generate PS10 submission", value=True)
        st.markdown(
            "_Note: disabling visualisations and PS10 packaging can significantly reduce runtime._"
        )

        if st.button("Start Stage 1 Run"):
            # Set up logger interceptor
            log_queue = queue.Queue()
            handler = QueueLogHandler(log_queue)
            handler.setLevel(logging.INFO)
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            
            # Start background thread
            outputs = {}
            t = threading.Thread(
                target=run_in_thread,
                args=(self.pipeline_service.run_stage1, (), {
                    "aoi_name": selected_aoi,
                    "compute_probabilities": compute_probabilities,
                    "generate_visualisations": generate_visualisations,
                    "generate_ps10_submission": generate_ps10_submission
                }, outputs)
            )
            t.start()
            
            st.info(f"Running Stage 1 change detection pipeline for {selected_aoi}...")
            
            log_placeholder = st.empty()
            progress_bar = st.progress(0.0)
            log_lines = []
            start_time = time.time()
            
            while t.is_alive():
                # Fetch any logs
                while not log_queue.empty():
                    log_lines.append(log_queue.get())
                
                # Show logs (last 30 lines to fit dashboard area)
                if log_lines:
                    log_placeholder.code("\n".join(log_lines[-30:]))
                
                # Update progress bar (simulated estimate based on average runtime)
                elapsed = time.time() - start_time
                progress_estimate = min(elapsed / 25.0, 0.95)
                progress_bar.progress(progress_estimate)
                
                time.sleep(0.2)
            
            # Capture any remaining log records
            while not log_queue.empty():
                log_lines.append(log_queue.get())
            if log_lines:
                log_placeholder.code("\n".join(log_lines))
            progress_bar.progress(1.0)
            
            # Remove logging interceptor
            root_logger.removeHandler(handler)
            
            # Render completion status
            if outputs.get("status") == "success":
                result = outputs["result"]
                st.success(f"Stage 1 completed successfully! Run ID: `{result.run_id}`")
                
                # Execution Summary Section
                st.markdown("### Execution Summary")
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                objects_count = len(result.object_records)
                change_area = result.prediction_summary.get("change_area_ha", 0.0)
                change_ratio = result.prediction_summary.get("change_ratio", 0.0)
                
                with col_sum1:
                    st.metric("Objects Detected", f"{objects_count} components")
                with col_sum2:
                    st.metric("Change Area (ha)", f"{change_area:.2f} ha")
                with col_sum3:
                    st.metric("Change Ratio", f"{change_ratio * 100:.2f}%")
                
                # Deliverables table
                st.markdown("#### Exported Deliverables")
                run_export_files = self.output_service.list_export_files(result.run_id)
                if run_export_files:
                    for path in run_export_files:
                        col_name, col_sz, col_dl = st.columns([5, 2, 2])
                        col_name.write(f"📄 `{path.name}`")
                        col_sz.write(human_readable_size(path.stat().st_size))
                        col_dl.download_button(
                            label="Download",
                            data=path.read_bytes(),
                            file_name=path.name,
                            key=f"run1-dl-{result.run_id}-{path.name}"
                        )
                else:
                    st.warning("No export files generated.")
            else:
                st.error(f"Stage 1 pipeline run failed: {outputs.get('error')}")

        st.divider()
        st.subheader("Run Stage 3 Batch Project")
        project_id = st.text_input("Project ID", value=f"project_{selected_aoi}")
        run_stage2 = st.checkbox("Run Stage 2 classification for each AOI", value=True)
        compute_shap = st.checkbox("Compute SHAP feature attributions (optional)", value=False)
        continue_on_error = st.checkbox("Continue on error", value=True)

        if st.button("Start Stage 3 Batch Run"):
            # Set up logger interceptor
            log_queue = queue.Queue()
            handler = QueueLogHandler(log_queue)
            handler.setLevel(logging.INFO)
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            
            # Start background thread
            outputs = {}
            t = threading.Thread(
                target=run_in_thread,
                args=(self.pipeline_service.run_stage3, (), {
                    "project_id": project_id or None,
                    "aoi_names": aoi_names,
                    "run_stage2_classification": run_stage2,
                    "compute_shap": compute_shap,
                    "continue_on_error": continue_on_error
                }, outputs)
            )
            t.start()
            
            st.info("Running Stage 3 batch project pipeline...")
            
            log_placeholder = st.empty()
            progress_bar = st.progress(0.0)
            log_lines = []
            start_time = time.time()
            
            while t.is_alive():
                # Fetch any logs
                while not log_queue.empty():
                    log_lines.append(log_queue.get())
                
                # Show logs (last 30 lines)
                if log_lines:
                    log_placeholder.code("\n".join(log_lines[-30:]))
                
                # Update progress bar
                elapsed = time.time() - start_time
                progress_estimate = min(elapsed / 45.0, 0.95)
                progress_bar.progress(progress_estimate)
                
                time.sleep(0.2)
            
            # Capture any remaining log records
            while not log_queue.empty():
                log_lines.append(log_queue.get())
            if log_lines:
                log_placeholder.code("\n".join(log_lines))
            progress_bar.progress(1.0)
            
            # Remove logging interceptor
            root_logger.removeHandler(handler)
            
            # Render completion status
            if outputs.get("status") == "success":
                result = outputs["result"]
                st.success(f"Stage 3 completed successfully! Project ID: `{result.project_id}`")
                
                # Execution Summary Section
                st.markdown("### Aggregated Project Summary")
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                
                tot_aois = result.project_summary.get("total_aois_processed", 0)
                tot_area = result.project_summary.get("total_change_area_ha", 0.0)
                tot_objs = result.project_summary.get("total_significant_objects", 0)
                
                with col_sum1:
                    st.metric("AOIs Processed", f"{tot_aois} areas")
                with col_sum2:
                    st.metric("Total Change Area", f"{tot_area:.2f} ha")
                with col_sum3:
                    st.metric("Total Objects Detected", f"{tot_objs} components")
                
                # Show project report preview
                report_file = result.output_dir / "project_report.txt" if result.output_dir else None
                if report_file and report_file.exists():
                    with st.expander("Project summary report"):
                        st.text(report_file.read_text(encoding="utf-8"))
            else:
                st.error(f"Stage 3 batch run failed: {outputs.get('error')}")


run_pipeline_page = RunPipelinePage()
