import streamlit as st


def metric_card(label: str, value: str, delta: str = "", help_text: str = "") -> None:
    cols = st.columns([2, 1])
    cols[0].markdown(f"**{label}**")
    cols[1].metric(label="", value=value, delta=delta)
    if help_text:
        st.caption(help_text)


def info_card(title: str, description: str) -> None:
    st.markdown(f"**{title}**")
    st.info(description)
