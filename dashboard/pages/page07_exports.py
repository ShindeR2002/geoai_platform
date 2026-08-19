import mimetypes

import streamlit as st
from dashboard.services.config_service import ConfigService
from dashboard.services.outputs_service import OutputService
from dashboard.utils.file_utils import human_readable_size


class ExportsPage:
    def __init__(self) -> None:
        self.config_service = ConfigService()
        self.output_service = OutputService(
            self.config_service.get_export_base_dir().as_posix()
        )

    def render(self) -> None:
        st.title("Exports")
        st.markdown(
            "Download available pipeline exports and review the files generated "
            "for each output run."
        )

        run_ids = self.output_service.list_runs(include_stage2=True)
        if not run_ids:
            st.warning("No completed pipeline runs available.")
            return

        run_id = st.selectbox("Select a run", run_ids)
        export_files = self.output_service.list_export_files(run_id)
        if not export_files:
            st.warning("No export files available for this run.")
            return

        st.subheader("Available export files")
        for path in export_files:
            cols = st.columns([5, 1, 1])
            cols[0].write(path.name)
            cols[1].write(human_readable_size(path.stat().st_size))
            cols[2].download_button(
                label="Download",
                data=path.read_bytes(),
                file_name=path.name,
                mime=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                key=f"export-{run_id}-{path.name}",
            )


exports_page = ExportsPage()
