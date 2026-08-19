"""Dashboard pages package."""

from .page01_home import home_page
from .page02_run_pipeline import run_pipeline_page
from .page03_map_viewer import map_viewer_page
from .page04_object_explorer import object_explorer_page
from .page05_statistics import statistics_page
from .page06_compare_page import compare_page
from .page07_exports import exports_page
from .page08_settings import settings_page
from .page09_research import ResearchPage
from .page10_validation import ValidationPage
from .page11_datasets import DatasetExplorerPage
from .page12_benchmarks import BenchmarkCenterPage
from .page13_research_center import ResearchCenterPage
from .page14_transformers import TransformerBenchmarkPage

research_page = ResearchPage()
validation_page = ValidationPage()
datasets_page = DatasetExplorerPage()
benchmarks_page = BenchmarkCenterPage()
research_center_page = ResearchCenterPage()
transformer_benchmark_page = TransformerBenchmarkPage()

__all__ = [
    "home_page",
    "run_pipeline_page",
    "map_viewer_page",
    "object_explorer_page",
    "statistics_page",
    "compare_page",
    "exports_page",
    "settings_page",
    "research_page",
    "validation_page",
    "datasets_page",
    "benchmarks_page",
    "research_center_page",
    "transformer_benchmark_page",
]
