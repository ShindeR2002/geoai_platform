import streamlit as st
from pathlib import Path
from dashboard.pages import (
    home_page,
    run_pipeline_page,
    map_viewer_page,
    object_explorer_page,
    statistics_page,
    compare_page,
    exports_page,
    settings_page,
    research_page,
    validation_page,
    datasets_page,
    benchmarks_page,
    research_center_page,
    transformer_benchmark_page,
)

PAGE_OPTIONS = [
    "Home",
    "Run Pipeline",
    "Map Viewer",
    "Object Explorer",
    "Statistics",
    "Compare AOIs",
    "Exports",
    "Research & Experiments",
    "Scientific Validation",
    "Public Dataset Explorer",
    "Benchmark Center",
    "Research Center",
    "Model Comparison Explorer",
    "Settings",
]


from dashboard.utils.styles import inject_custom_css


def main() -> None:
    st.set_page_config(
        page_title="GeoAI Platform Dashboard",
        page_icon="🛰️",
        layout="wide",
    )
    inject_custom_css()

    st.sidebar.title("GeoAI Platform")
    st.sidebar.markdown("Build, inspect, and export geospatial results from the pipeline.")
    selection = st.sidebar.radio("Navigation", PAGE_OPTIONS)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Outputs root:** `outputs/`\n\n**Configs:** `configs/`\n\n**Dashboard:** `dashboard/`")

    if selection == "Home":
        home_page.render()
    elif selection == "Run Pipeline":
        run_pipeline_page.render()
    elif selection == "Map Viewer":
        map_viewer_page.render()
    elif selection == "Object Explorer":
        object_explorer_page.render()
    elif selection == "Statistics":
        statistics_page.render()
    elif selection == "Compare AOIs":
        compare_page.render()
    elif selection == "Exports":
        exports_page.render()
    elif selection == "Research & Experiments":
        research_page.render()
    elif selection == "Scientific Validation":
        validation_page.render()
    elif selection == "Public Dataset Explorer":
        datasets_page.render()
    elif selection == "Benchmark Center":
        benchmarks_page.render()
    elif selection == "Research Center":
        research_center_page.render()
    elif selection == "Model Comparison Explorer":
        transformer_benchmark_page.render()
    elif selection == "Settings":
        settings_page.render()


if __name__ == "__main__":
    main()
