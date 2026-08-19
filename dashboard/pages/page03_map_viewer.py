import json
import base64
from io import BytesIO
from pathlib import Path
from PIL import Image
import numpy as np
import rasterio
import folium
import streamlit as st
from streamlit_folium import st_folium

from dashboard.services.config_service import ConfigService
from dashboard.services.outputs_service import OutputService


def get_raster_overlay(tif_path: Path):
    """Read a GeoTIFF and convert it to a transparent RGBA PNG base64 string for Folium overlay."""
    try:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            data = src.read(1)
            
        h, w = data.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Set pixels where change occurs (1) to transparent red
        change_pixels = (data == 1)
        rgba[change_pixels] = [235, 94, 85, 140]  # sunset red, alpha=140
        
        img = Image.fromarray(rgba, "RGBA")
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        folium_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
        return f"data:image/png;base64,{img_str}", folium_bounds
    except Exception as e:
        st.sidebar.warning(f"Failed to process raster overlay: {e}")
        return None, None


class MapViewerPage:
    def __init__(self) -> None:
        self.config_service = ConfigService()
        self.output_service = OutputService(
            self.config_service.get_export_base_dir().as_posix()
        )

    def render(self) -> None:
        st.title("Map Viewer")
        st.markdown(
            "Inspect change object geometries and exported GeoJSON directly in the browser."
        )

        run_ids = self.output_service.list_runs(include_stage2=True)
        if not run_ids:
            st.warning("No pipeline runs found in outputs.")
            return

        run_id = st.selectbox("Select a run", run_ids)
        geojson_path = self.output_service.find_geojson(run_id)
        if not geojson_path:
            st.warning("No GeoJSON export available for this run.")
            return

        st.markdown(f"**GeoJSON:** `{geojson_path.name}`")
        try:
            data = json.loads(geojson_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Unable to read GeoJSON export: {exc}")
            return

        map_object = folium.Map(location=(0.0, 0.0), zoom_start=2, tiles="CartoDB positron")
        
        # Add prediction raster overlay if GeoTIFF exists
        tif_path = self.output_service.find_export_file(run_id, ".tif")
        raster_bounds = None
        if tif_path:
            img_data, r_bounds = get_raster_overlay(tif_path)
            if img_data and r_bounds:
                raster_bounds = r_bounds
                folium.raster_layers.ImageOverlay(
                    image=img_data,
                    bounds=raster_bounds,
                    name="Prediction Raster (Red Overlay)",
                    opacity=0.8,
                    interactive=False,
                    zindex=1
                ).add_to(map_object)

        # Setup tooltip/popup attributes
        fields = []
        aliases = []
        if data.get("features"):
            first_props = data["features"][0].get("properties", {})
            potential_fields = ["object_id", "area_m2", "class", "confidence", "compact", "perimeter", "method"]
            for field in potential_fields:
                if field in first_props:
                    fields.append(field)
                    aliases.append(field.replace("_", " ").title() + ":")

        # Add GeoJSON layer
        geojson_layer = folium.GeoJson(
            data,
            name="Change Objects (Polygons)",
            tooltip=folium.GeoJsonTooltip(fields=fields[:2], aliases=aliases[:2]) if len(fields) >= 2 else None,
            popup=folium.GeoJsonPopup(fields=fields, aliases=aliases) if fields else None,
            style_function=lambda x: {
                "fillColor": "#ab7df8",
                "color": "#8957e5",
                "weight": 2,
                "fillOpacity": 0.4
            }
        )
        geojson_layer.add_to(map_object)
        
        # Fit bounds
        bounds = geojson_layer.get_bounds()
        if bounds and len(bounds) == 2 and all(corner is not None for corner in bounds) and all(value is not None for corner in bounds for value in corner):
            map_object.fit_bounds(bounds)
        elif raster_bounds:
            map_object.fit_bounds(raster_bounds)
            
        folium.LayerControl().add_to(map_object)
        st_folium(map_object, height=650, use_container_width=True)

        figures = self.output_service.list_figures(run_id)
        if figures:
            st.subheader("Change Overlay")
            for figure in figures:
                st.image(str(figure), caption=figure.name, use_container_width=True)


map_viewer_page = MapViewerPage()
