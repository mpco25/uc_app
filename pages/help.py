# pages/Help.py

from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Help",
    layout="wide",
)

st.title("Unity Catalog Metadata Assistant - Help")

help_file = Path("docs/help.md")

if help_file.exists():
    st.markdown(
        help_file.read_text(
            encoding="utf-8"
        )
    )
else:
    st.error("docs/help.md not found")