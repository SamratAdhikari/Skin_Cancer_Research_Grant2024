"""
app.py
------
Streamlit front-end for the hierarchical, multimodal skin-lesion classifier.

Run with:
    streamlit run app.py

The app loads the three trained ``.keras`` models from ``models/`` and runs
inference locally (Stage 1 benign/malignant -> Stage 2 subtype) with a Grad-CAM
explanation. If no local models are present it falls back to the remote API
configured via a ``.env`` file (API_URL / HF_API_TOKEN).
"""

import os

import streamlit as st
from dotenv import load_dotenv

from components.about import show_about_section
from components.header import show_header
from components.image_upload import show_image_upload
from components.results import show_results


def _load_css() -> None:
    css_path = os.path.join("static", "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(
        page_title="Skin Cancer Detection", page_icon="🔬", layout="wide"
    )
    _load_css()

    show_header()

    col1, col2 = st.columns([1, 1])
    with col1:
        show_image_upload()
    with col2:
        show_results()

    show_about_section()


if __name__ == "__main__":
    main()
