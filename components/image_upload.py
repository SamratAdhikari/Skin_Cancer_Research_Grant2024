import numpy as np
import streamlit as st
from PIL import Image

from utils.local_infer import LOCALIZATIONS, SEXES, load_pipeline, run_analysis


@st.cache_resource(show_spinner=False)
def _get_pipeline():
    """Load the local hierarchical pipeline once and cache it across reruns."""
    return load_pipeline()


def show_image_upload():
    """Image upload + patient-metadata form, then run local multimodal inference."""
    st.markdown("### Upload an image")
    st.write("Upload a high-quality dermatoscopic image of the skin lesion.")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    # ---- Patient metadata (drives the multimodal branch of every stage) ----
    st.markdown("#### Patient information")
    st.caption(
        "Optional but recommended — patient context (age, sex, body site) is a "
        "real risk factor and feeds the model's metadata branch."
    )
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        age_known = st.checkbox("Age known", value=True)
        age = st.slider("Age", 0, 100, 50, disabled=not age_known)
        age_val = float(age) if age_known else None
    with mcol2:
        sex = st.selectbox("Sex", SEXES, index=0)
        localization = st.selectbox("Anatomical site", LOCALIZATIONS, index=0)

    if uploaded_file is not None:
        status = _get_pipeline()

        analyze_button = st.button("Analyze Image", type="primary")
        if analyze_button:
            image = Image.open(uploaded_file).convert("RGB")
            if not status.available:
                # Graceful fallback to the remote API if it is configured.
                _run_remote_fallback(image, status.error)
            else:
                with st.spinner("Running hierarchical multimodal analysis..."):
                    try:
                        result, overlay = run_analysis(
                            status.pipeline,
                            image,
                            age=age_val,
                            sex=sex,
                            localization=localization,
                            gradcam=True,
                        )
                        st.session_state.result = result
                        st.session_state.gradcam = (
                            np.asarray(overlay) if overlay is not None else None
                        )
                        st.session_state.uploaded_image = image
                        st.success("Analysis complete!")
                    except Exception as exc:  # keep the UI alive on any failure
                        st.error(f"Unexpected error during analysis: {exc}")

        if analyze_button or "result" in st.session_state:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)


def _run_remote_fallback(image, load_error: str) -> None:
    """If local models are absent, try the original remote API (if configured)."""
    st.info(load_error)
    try:
        from utils.api import analyze_image
    except Exception:
        st.warning("Remote API client unavailable; cannot analyze without local models.")
        return

    with st.spinner("Contacting remote API..."):
        try:
            result, error = analyze_image(image)
            if error:
                st.error(f"Remote API error: {error}")
            else:
                result["source"] = "remote"
                st.session_state.result = result
                st.session_state.gradcam = None
                st.session_state.uploaded_image = image
                st.success("Analysis complete (remote API).")
        except Exception as exc:
            st.error(f"Remote API call failed: {exc}")
