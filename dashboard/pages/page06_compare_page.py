import json
from pathlib import Path
import streamlit as st
from dashboard.services.config_service import ConfigService
from dashboard.services.outputs_service import OutputService


class ComparePage:
    def __init__(self) -> None:
        self.config_service = ConfigService()
        self.output_service = OutputService(
            self.config_service.get_export_base_dir().as_posix()
        )

    def render(self) -> None:
        st.title("Compare AOIs")
        st.markdown(
            "Compare completed run summaries, validate exports, and generate mask comparison charts."
        )

        # Triggers panel
        st.subheader("Run New Validation & Comparison")
        all_runs = self.output_service.list_runs(include_stage2=True)
        
        if not all_runs:
            st.warning("No runs found in outputs to compare.")
        else:
            selected_runs = st.multiselect("Select runs to compare", all_runs, default=all_runs[:2] if len(all_runs) >= 2 else all_runs)
            
            if st.button("Generate Validation & Comparison"):
                if not selected_runs:
                    st.error("Please select at least one run.")
                else:
                    with st.spinner("Running validation & comparison checks..."):
                        try:
                            import sys
                            sys.path.insert(0, str(Path.cwd()))
                            from scripts.validate_and_compare import compare_results, generate_visualizations, validate_exports
                            
                            # Validate each selected run
                            for run_id in selected_runs:
                                report = validate_exports(run_id)
                                report_file = Path(f"outputs/{run_id}/VALIDATION_REPORT.md")
                                report_file.write_text(report.to_markdown(), encoding="utf-8")
                                
                            # Compare metrics
                            comparison = compare_results(selected_runs)
                            comp_file = Path("outputs/COMPARISON.json")
                            with open(comp_file, "w") as f:
                                json.dump(comparison, f, indent=2)
                                
                            # Create visuals
                            generate_visualizations(selected_runs)
                            
                            st.success("Comparison files generated successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Execution failed: {e}")

        st.divider()
        st.subheader("Latest Comparison Data")
        comparison = self.output_service.get_comparison_data()
        
        if not comparison:
            st.warning("No comparison data found in outputs/COMPARISON.json. Use the controls above to generate one.")
            return

        st.write(f"**Last Comparison Run Timestamp:** `{comparison.get('timestamp')}`")

        runs = comparison.get("runs", {})
        if not runs:
            st.warning("Comparison file contains no run data.")
            return

        # Show comparison statistics in table
        st.table(
            [
                {
                    "Run ID": run_id,
                    "Total objects": stats.get("total_objects", 0),
                    "Total area (ha)": f"{stats.get('total_area_ha', 0):.2f}" if stats.get('total_area_ha') else "0.00",
                    "Mean area (ha)": f"{stats.get('mean_area_ha', 0):.4f}" if stats.get('mean_area_ha') else "0.0000",
                    "Max area (ha)": f"{stats.get('max_area_ha', 0):.2f}" if stats.get('max_area_ha') else "0.00",
                }
                for run_id, stats in runs.items()
            ]
        )

        image_path = self.output_service.get_comparison_image()
        if image_path:
            st.subheader("Comparison Mask Image")
            st.image(str(image_path), caption="Side-by-side binary change mask comparison visualizer", use_container_width=True)


compare_page = ComparePage()
