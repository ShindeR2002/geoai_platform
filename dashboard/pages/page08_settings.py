import streamlit as st
from pathlib import Path
import pandas as pd

from dashboard.services.config_service import ConfigService
from dashboard.utils.file_utils import read_json_file


class SettingsPage:
    def __init__(self) -> None:
        self.config_service = ConfigService()

    def render(self) -> None:
        st.title("Settings")
        st.markdown(
            "Review dashboard configuration, pipeline settings, and model metadata."
        )

        config = self.config_service.get_platform_config()
        
        tab_aoi, tab_pipe, tab_model, tab_feature = st.tabs([
            "🛰️ AOI Configurations", 
            "⚙️ Pipeline & Export Settings", 
            "🧠 Model Information", 
            "📋 Feature Metadata"
        ])
        
        # Tab 1: AOI Configuration
        with tab_aoi:
            st.subheader("Configured Areas of Interest (AOIs)")
            aoi_data = []
            for aoi in config.processing.aois:
                aoi_data.append({
                    "AOI Name": aoi.name,
                    "T1 Year": aoi.t1_year,
                    "T2 Year": aoi.t2_year,
                    "EO Sensor": aoi.sensor_eo,
                    "SAR Sensor": aoi.sensor_sar or "None",
                    "Expected CRS": aoi.crs or "Not Set",
                    "Resolution (m)": f"{aoi.resolution_m:.1f} m",
                    "Index Mode": aoi.index_mode,
                    "Ref Latitude": aoi.reference_lat or "N/A",
                    "Ref Longitude": aoi.reference_lon or "N/A",
                    "Data Directory": aoi.data_dir
                })
            df_aoi = pd.DataFrame(aoi_data)
            st.dataframe(df_aoi, use_container_width=True)
            st.caption(f"Currently monitoring {len(df_aoi)} active AOIs.")

        # Tab 2: Pipeline & Export Settings
        with tab_pipe:
            st.subheader("Processing parameters")
            st.write(f"**Minimum Change Object Size:** `{config.processing.min_object_size_px} pixels` (regions below this size are filtered out)")
            st.write(f"**Contrast Stretch Limits:** `Low: {config.processing.contrast_stretch_low_pct}% | High: {config.processing.contrast_stretch_high_pct}%` (used for figures visualization)")
            
            st.markdown("**Composite Pseudo-Label Weights:**")
            w = config.processing.pseudo_labels.weights
            st.json({
                "SAR weight": w.sar,
                "NDVI weight": w.ndvi,
                "NDBI weight": w.ndbi,
                "NDWI weight": w.ndwi,
                "Sigma multiplier (threshold_sigma)": config.processing.pseudo_labels.threshold_sigma
            })
            
            st.subheader("Export parameters")
            st.json({
                "output_base_dir": config.export.output_base_dir,
                "geotiff_compress": config.export.geotiff_compress,
                "shapefile_encoding": config.export.shapefile_encoding,
                "ps10_submission_prefix": config.export.ps10_submission_prefix,
                "include_geojson": config.export.include_geojson,
                "include_csv": config.export.include_csv,
            })

            st.subheader("Dashboard settings")
            st.json({
                "project_manifest_dir": config.dashboard.project_manifest_dir,
                "default_aoi": config.dashboard.default_aoi,
                "tile_format": config.dashboard.tile_format,
            })

        # Tab 3: Model Information
        with tab_model:
            st.subheader("Random Forest Model Metadata")
            metadata = self.config_service.get_model_metadata()
            
            if metadata:
                st.write(f"**Model ID:** `{metadata.get('model_id')}`")
                st.write(f"**Model Class:** `{metadata.get('model_class')}`")
                st.write(f"**Estimator Count (n_estimators):** `{metadata.get('n_estimators')}`")
                st.write(f"**Number of Input Features:** `{metadata.get('n_features_in')}`")
                st.write(f"**Model Source:** `{metadata.get('source')}`")
                st.write(f"**Migration Date:** `{metadata.get('migration_date')}`")
                st.write(f"**Registered MD5 Checksum:** `{metadata.get('model_file_md5')}`")
                
                st.markdown("**Complete metadata details:**")
                st.json(metadata)
            else:
                st.warning(
                    f"Model metadata file not found at: {Path(config.model.metadata_path)}"
                )
                
            st.markdown("---")
            st.write(f"**Model Weights File Path:** `{config.model.model_path}`")
            st.write(f"**Expected Input Dimension:** `{config.model.expected_feature_count} features`")
            st.write(f"**Class Weight Strategy:** `{config.model.class_weight or 'None (Uniform)'}`")

        # Tab 4: Feature Metadata
        with tab_feature:
            st.subheader("Model Feature Dimension Mapping")
            st.markdown(
                "The feature lineage order below is frozen by the production Random Forest classifier. "
                "Any modifications will mismatch input array dimensions during model inference."
            )
            
            feature_names = self.config_service.get_feature_names()
            df_feat = pd.DataFrame([
                {"Feature Index": idx, "Feature Name": name, "Epoch": "T1 (2021)" if "2021" in name else ("T2 (2024)" if "2024" in name else "Delta / Shift")}
                for idx, name in enumerate(feature_names)
            ])
            st.dataframe(df_feat, use_container_width=True)
            st.caption(f"Model expects exactly {len(df_feat)} input features.")


settings_page = SettingsPage()
