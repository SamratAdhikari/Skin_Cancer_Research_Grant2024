import pandas as pd
import plotly.express as px
import streamlit as st


def show_results():
    """Display the analysis results (local hierarchical pipeline or remote API)."""
    st.markdown("### Analysis Results")

    result = st.session_state.get("result")
    if result is None:
        _show_placeholder()
        _show_disclaimer()
        return

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    if "stage1_label" in result:
        _render_local(result)
    else:
        _render_remote(result)  # backwards-compatible with the old API schema
    st.markdown("</div>", unsafe_allow_html=True)

    _show_disclaimer()


# ------------------------------------------------------------------ local schema
def _render_local(result: dict) -> None:
    is_mal = result.get("is_malignant", False)
    s1_conf = result.get("stage1_confidence", 0) or 0

    if is_mal:
        st.markdown(f"<h2>🔴 {result['stage1_display']} (Stage 1)</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p><strong>Important:</strong> the model flags this lesion as "
            "potentially <strong>malignant</strong>. Please consult a "
            "dermatologist promptly.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<h2>🟡 {result['stage1_display']} (Stage 1)</h2>", unsafe_allow_html=True)
        st.markdown(
            "<p>The model suggests a <strong>benign</strong> (non-cancerous) "
            "lesion. This is not a diagnosis — monitor for change and consult a "
            "professional if concerned.</p>",
            unsafe_allow_html=True,
        )

    _confidence_meter("Benign vs. malignant confidence", s1_conf)

    # ---- Stage 2 subtype ----
    subtype_display = result.get("subtype_display")
    if subtype_display:
        st.markdown(
            f"#### Subtype: {subtype_display} "
            f"({result.get('subtype_confidence', 0):.1f}%)"
        )
        probs = result.get("subtype_probs") or {}
        if probs:
            _subtype_bar(probs)

    # ---- Explainability: Grad-CAM ----
    overlay = st.session_state.get("gradcam")
    if overlay is not None:
        st.markdown("#### Where the model looked (Grad-CAM)")
        st.image(
            overlay,
            caption="Warmer regions drove the Stage-1 decision. Check that the "
            "model attends to the lesion, not to hair, ruler marks or ink.",
            use_container_width=True,
        )

    # ---- Metadata note ----
    if result.get("used_metadata"):
        st.caption("✓ Prediction used patient metadata (age, sex, anatomical site).")
    else:
        st.caption("ⓘ Image-only prediction (no patient metadata was applied).")


def _confidence_meter(label: str, value: float) -> None:
    st.markdown(f"<p>{label}: <strong>{value:.1f}%</strong></p>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="confidence-meter">
            <div class="confidence-value" style="width: {max(0, min(100, value))}%;">
                {value:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _subtype_bar(prob_dict: dict) -> None:
    data = pd.DataFrame(
        {"Category": list(prob_dict.keys()), "Confidence": list(prob_dict.values())}
    ).sort_values("Confidence", ascending=False)

    fig = px.bar(
        data,
        x="Category",
        y="Confidence",
        color="Confidence",
        color_continuous_scale=["#e9ecef", "#4e54c8"],
        text="Confidence",
        labels={"Confidence": "Confidence (%)", "Category": "Classification"},
        height=300,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------ remote API (legacy) schema
def _render_remote(result: dict) -> None:
    """Render the original external-API response shape (kept as a fallback)."""
    if result.get("LesionDetected") is False:
        st.markdown(f"<h2>🟢 {result.get('PredictedClass')}</h2>", unsafe_allow_html=True)
        _confidence_meter("Confidence", float(result.get("Confidence", 0) or 0))
        st.markdown(
            "<p>No further action indicated by this analysis; consult a "
            "professional if you have concerns.</p>",
            unsafe_allow_html=True,
        )
        return

    lesion_confidence = float(result.get("Confidence", 0) or 0)
    if result.get("MalignantPredicted") is True:
        st.markdown(
            f"<h2>🔴 {result.get('MalignantPredictedClass', 'Malignant')}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h2>🟡 {result.get('BenignPredictedClass', 'Benign')}</h2>",
            unsafe_allow_html=True,
        )
    _confidence_meter("Lesion detection confidence", lesion_confidence)

    if "SubClassification" in result:
        st.markdown("#### Detailed Classification")
        _subtype_bar(result["SubClassification"])


# ------------------------------------------------------------------------ shared
def _show_placeholder() -> None:
    st.markdown(
        """
        <div class="small-result-card" style="text-align: center; padding: 20px;">
            <img src="https://img.icons8.com/fluency/96/000000/upload-to-cloud.png"
                 style="opacity: 0.5; width: 30px; height: 30px;">
            <p style="color: #666; margin-top: 10px; font-size: 12px;">
                Upload an image and click "Analyze Image" to see results here.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            ⚠️ <strong>Medical Disclaimer:</strong> This tool is a research
            prototype for educational purposes only and is not a medical device.
            It must not be used for diagnosis. Always consult a qualified
            healthcare provider.
        </div>
        """,
        unsafe_allow_html=True,
    )
