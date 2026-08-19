import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional


def render_table(csv_path: Path, max_rows: int = 10) -> None:
    if not csv_path.exists():
        st.warning("CSV file not found.")
        return
    try:
        df = pd.read_csv(csv_path)
        st.dataframe(df.head(max_rows), use_container_width=True)
        if len(df) > max_rows:
            st.caption(f"Showing first {max_rows} rows of {len(df)}.")
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
