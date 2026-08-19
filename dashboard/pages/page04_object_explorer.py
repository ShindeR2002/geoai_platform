import pandas as pd
import streamlit as st
from dashboard.services.config_service import ConfigService
from dashboard.services.outputs_service import OutputService


class ObjectExplorerPage:
    def __init__(self) -> None:
        self.config_service = ConfigService()
        self.output_service = OutputService(
            self.config_service.get_export_base_dir().as_posix()
        )

    def render(self) -> None:
        st.title("Object Explorer")
        st.markdown(
            "Browse per-object statistics exported from Stage 1 and Stage 2 pipeline runs."
        )

        run_ids = self.output_service.list_runs(include_stage2=True)
        if not run_ids:
            st.warning("No completed pipeline runs detected in outputs.")
            return

        run_id = st.selectbox("Select a run", run_ids)
        csv_by_aoi = {
            aoi_name: csv_path
            for aoi_name in self.config_service.get_aoi_names()
            if (csv_path := self.output_service.find_csv(run_id, aoi_name=aoi_name))
        }
        if not csv_by_aoi:
            st.warning("No object statistics CSV file found for this run.")
            return

        aoi_name = st.selectbox("Select AOI", list(csv_by_aoi))
        csv_path = csv_by_aoi[aoi_name]

        st.markdown(f"**CSV:** `{csv_path.name}`")
        
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                st.warning("Selected CSV file contains no object records.")
                return
            
            # Interactive Filter Panel
            st.subheader("Filter and Sort Objects")
            
            col_f1, col_f2, col_f3 = st.columns(3)
            
            # Class filter
            with col_f1:
                if "class" in df.columns:
                    classes = ["All"] + sorted(list(df["class"].dropna().unique()))
                    selected_class = st.selectbox("Class Filter", classes)
                    if selected_class != "All":
                        df = df[df["class"] == selected_class]
                else:
                    st.info("Class labels not available in this stage.")
                    
            # Confidence filter
            with col_f2:
                if "confidence" in df.columns and len(df) > 0:
                    min_conf = float(df["confidence"].min())
                    max_conf = float(df["confidence"].max())
                    if min_conf == max_conf:
                        st.info(f"Constant confidence score: {min_conf:.3f}")
                    else:
                        conf_range = st.slider("Confidence Score Range", min_conf, max_conf, (min_conf, max_conf))
                        df = df[(df["confidence"] >= conf_range[0]) & (df["confidence"] <= conf_range[1])]
                else:
                    st.info("Confidence scores not available.")
            
            # Area filter
            with col_f3:
                area_col = "area_m2" if "area_m2" in df.columns else ("area_px" if "area_px" in df.columns else None)
                if area_col and len(df) > 0:
                    min_area = float(df[area_col].min())
                    max_area = float(df[area_col].max())
                    if min_area == max_area:
                        st.info(f"Constant Area: {min_area:.1f} m²")
                    else:
                        area_range = st.slider(f"Area Range ({'m²' if area_col == 'area_m2' else 'px'})", min_area, max_area, (min_area, max_area))
                        df = df[(df[area_col] >= area_range[0]) & (df[area_col] <= area_range[1])]
                        
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                # Text/ID Search
                search_query = st.text_input("Search by Object ID", "").strip()
                if search_query:
                    try:
                        search_id = int(search_query)
                        if "object_id" in df.columns:
                            df = df[df["object_id"] == search_id]
                    except ValueError:
                        st.error("Please enter a valid integer for Object ID search.")
            
            with col_s2:
                # Sort column
                sortable_cols = [c for c in df.columns if c in ["object_id", "area_px", "area_m2", "confidence", "perimeter", "compact"]]
                sort_col = st.selectbox("Sort Column", sortable_cols if sortable_cols else list(df.columns))
            
            with col_s3:
                # Sort order
                sort_order = st.selectbox("Sort Order", ["Ascending", "Descending"])
                if sort_col in df.columns:
                    df = df.sort_values(by=sort_col, ascending=(sort_order == "Ascending"))

            # Metrics for Filtered List
            st.markdown("---")
            st.subheader("Filtered Subset Summary")
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                st.metric("Total Count", f"{len(df)} objects")
            with col_m2:
                avg_conf = df["confidence"].mean() if "confidence" in df.columns and len(df) > 0 else 0.0
                st.metric("Average Confidence", f"{avg_conf:.3f}" if avg_conf > 0 else "N/A")
            with col_m3:
                tot_area = df[area_col].sum() if area_col and len(df) > 0 else 0.0
                unit = "ha" if area_col == "area_m2" else "px"
                # If area is in m2, format to hectares for readability
                display_area = f"{tot_area / 10000.0:.2f} ha" if area_col == "area_m2" else f"{tot_area:,.0f} px"
                st.metric("Aggregate Area", display_area if tot_area > 0 else "N/A")

            # Table display
            st.markdown("#### Object Records Table")
            st.dataframe(df, use_container_width=True)
            st.caption(f"Showing {len(df)} matching objects. Columns are sortable.")

            # Filtered CSV Export
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Export filtered dataset ({len(df)} objects)",
                data=csv_data,
                file_name=f"Filtered_Objects_{aoi_name}_{run_id}.csv",
                mime="text/csv",
                key=f"export-filtered-csv-{run_id}-{aoi_name}",
            )
            
        except Exception as exc:
            st.error(f"Unable to explore objects: {exc}")


object_explorer_page = ObjectExplorerPage()
