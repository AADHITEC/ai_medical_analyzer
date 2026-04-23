"""
app.py  —  AI Medical Image Analyzer
Streamlit application with:
  • OpenCV preprocessing pipeline
  • ResNet-50 chest X-ray classification
  • Grad-CAM visualization
  • FAISS RAG engine
  • GPT-4 clinical explanations
  • PDF report download

FIXES APPLIED:
  1. White box text color fixed (dark blue)
  2. Button text color fixed (white)
  3. st.image() use_container_width → use_column_width (Streamlit compatibility)
  4. faiss import wrapped in try/except with helpful error message
  5. Drop zone text color fixed (readable dark colors)
"""

import os
import sys
import tempfile
import time

import streamlit as st
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(ROOT, "models"),
    os.path.join(ROOT, "rag"),
    os.path.join(ROOT, "utils"),
])

load_dotenv(os.path.join(ROOT, ".env"))

# ── Lazy imports (avoid crashing if GPU libs not ready) ───────────────────────
@st.cache_resource(show_spinner="Loading AI model…")
def load_analyzer():
    from classifier import MedicalImageAnalyzer
    model_path = os.path.join(ROOT, os.getenv("MODEL_PATH", "models/chest_xray_resnet50.pth"))
    return MedicalImageAnalyzer(model_path=model_path)


@st.cache_resource(show_spinner="Building RAG knowledge base…")
def load_rag(api_key: str):
    try:
        import faiss
    except ImportError:
        st.error(
            "❌ **faiss-cpu not installed!**\n\n"
            "Run this in your terminal:\n"
            "```\npip install faiss-cpu\n```\n"
            "Then restart the app."
        )
        st.stop()
    sys.path.insert(0, os.path.join(ROOT, "rag"))
    from rag_engine import MedicalRAGEngine
    return MedicalRAGEngine(openai_api_key=api_key)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Medical Image Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 2.2rem; font-weight: 700; color: #1A5276;
    margin-bottom: 0.2rem;
  }
  .subtitle {
    font-size: 1rem; color: #5D6D7E; margin-bottom: 1.5rem;
  }
  .result-card {
    padding: 1.2rem 1.5rem; border-radius: 10px;
    border-left: 5px solid; margin-bottom: 1rem;
  }
  .severity-none     { border-color: #27AE60; background: #EAFAF1; }
  .severity-moderate { border-color: #E67E22; background: #FEF9E7; }
  .severity-high     { border-color: #E74C3C; background: #FDEDEC; }
  .metric-box {
    background: #EBF5FB; border-radius: 8px;
    padding: 0.8rem 1rem; text-align: center;
    border: 1px solid #D6EAF8;
  }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: #1A5276; }
  .metric-label { font-size: 0.8rem; color: #5D6D7E; margin-top: 2px; }
  .source-chip {
    display: inline-block; padding: 4px 10px; border-radius: 15px;
    background: #EBF5FB; color: #1A5276; font-size: 0.78rem;
    margin: 2px; border: 1px solid #AED6F1;
  }
  .step-badge {
    display: inline-block; width: 26px; height: 26px;
    border-radius: 50%; background: #1A5276; color: white;
    text-align: center; line-height: 26px; font-size: 0.85rem;
    font-weight: 600; margin-right: 6px;
  }

  /* FIX 1: Button text always white */
  .stButton>button {
    background: #1A5276 !important; color: white !important; border: none;
    border-radius: 6px; padding: 0.5rem 1.5rem;
  }
  .stButton>button:hover { background: #154360 !important; color: white !important; }
  .stButton>button p,
  .stButton>button span,
  .stButton>button div {
    color: white !important;
  }

  /* FIX 2: White card text readable */
  [data-testid="stHorizontalBlock"] p,
  [data-testid="stHorizontalBlock"] span,
  [data-testid="stHorizontalBlock"] label,
  [data-testid="stHorizontalBlock"] b,
  [data-testid="stHorizontalBlock"] div {
    color: #1A5276 !important;
  }

  /* FIX 3: Download button text white */
  [data-testid="stDownloadButton"] button {
    color: white !important;
  }
  [data-testid="stDownloadButton"] button p,
  [data-testid="stDownloadButton"] button span {
    color: white !important;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/lung.png", width=64)
    st.markdown("## 🏥 AI Medical Analyzer")
    st.markdown("---")

    st.markdown("### ⚙️ Configuration")
    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Optional. Required only for GPT-4 explanations. Leave blank to use rule-based explanations.",
    )

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info(
        "**Architecture**: ResNet-50\n\n"
        "**Classes**: Normal, Pneumonia, COVID-19, TB, Pleural Effusion\n\n"
        "**Input**: 224×224 RGB\n\n"
        "**RAG**: FAISS + MiniLM-L6"
    )

    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        "This tool is for **research and educational** purposes only. "
        "All results must be validated by a licensed radiologist."
    )

    st.markdown("---")
    st.markdown("### 🔗 Resources")
    st.markdown(
        "- [NIH ChestX-ray14 Dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)\n"
        "- [WHO Radiology Guidelines](https://www.who.int/)\n"
        "- [COVID-RADS Classification](https://pubs.rsna.org/)\n"
        "- [GitHub Repository](#)"
    )


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏥 AI Medical Image Analyzer</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Deep Learning + OpenCV + RAG-Powered Chest X-ray Analysis</div>',
    unsafe_allow_html=True,
)

# Pipeline steps banner
cols = st.columns(5)
steps = [
    ("📤", "Upload X-ray"),
    ("🔬", "OpenCV Preprocessing"),
    ("🧠", "ResNet-50 Classification"),
    ("🗂️", "RAG Retrieval"),
    ("💬", "GPT-4 Explanation"),
]
for col, (icon, label) in zip(cols, steps):
    col.markdown(
        f'<div style="text-align:center;padding:8px;background:#EBF5FB;'
        f'border-radius:8px;font-size:0.85rem;color:#1A5276;">'
        f'{icon}<br><b style="color:#1A5276;">{label}</b></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📂 Upload a Chest X-ray image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Upload a PA or AP chest X-ray. Supported: JPG, PNG, BMP, TIFF",
)

if uploaded_file is None:
    st.markdown("""
    <div style="text-align:center;padding:3rem;background:#F8F9FA;
    border-radius:12px;border:2px dashed #AED6F1;">
      <h3 style="color:#1A5276;">📁 Drop a chest X-ray here to begin analysis</h3>
      <p style="color:#5D6D7E;">Supported formats: JPG, PNG, BMP, TIFF</p>
      <p style="color:#5D6D7E;">For best results, use standard PA or AP chest radiographs</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Save uploaded file to temp
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ── Analysis button ───────────────────────────────────────────────────────
    if st.button("🔍 Analyze X-ray", use_container_width=True):
        st.session_state["analysis_done"] = False

        with st.status("Running analysis pipeline...", expanded=True) as status:
            st.write("⚙️ Loading ResNet-50 model...")
            analyzer = load_analyzer()

            st.write("🔬 Running OpenCV preprocessing + model inference...")
            t0 = time.time()
            result = analyzer.analyze(tmp_path)
            inference_time = time.time() - t0
            result["inference_time"] = inference_time

            st.write("🗂️ Retrieving medical knowledge (RAG)...")
            rag = load_rag(openai_key)
            rag_result = rag.generate_explanation(
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                cv_metrics=result["cv_metrics"],
                class_probs=result["class_probs"],
            )

            st.session_state["result"]        = result
            st.session_state["rag_result"]    = rag_result
            st.session_state["tmp_path"]      = tmp_path
            st.session_state["analysis_done"] = True
            status.update(label="✅ Analysis complete!", state="complete")

    # ── Results display ───────────────────────────────────────────────────────
    if st.session_state.get("analysis_done"):
        result     = st.session_state["result"]
        rag_result = st.session_state["rag_result"]
        tmp_path   = st.session_state["tmp_path"]

        from helpers import (make_probability_chart, make_metrics_radar,
                              make_severity_gauge, get_severity_info,
                              generate_pdf_report, apply_clahe_display)

        sev_bg, sev_color, sev_msg = get_severity_info(result["severity"])
        pred = result["predicted_class"]
        conf = result["confidence"]

        # ── Top result banner ─────────────────────────────────────────────────
        st.markdown(
            f'<div class="result-card severity-{result["severity"]}">'
            f'<h2 style="margin:0;color:{sev_color};">{sev_msg}</h2>'
            f'<p style="margin:4px 0 0;font-size:1.1rem;color:#1A5276;">'
            f'<b>Predicted:</b> {pred} &nbsp;|&nbsp; '
            f'<b>Confidence:</b> {conf*100:.1f}% &nbsp;|&nbsp; '
            f'<b>Inference:</b> {result["inference_time"]*1000:.0f}ms</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Main columns: images + charts ─────────────────────────────────────
        img_col, chart_col = st.columns([1.2, 1.8])

        with img_col:
            tab1, tab2, tab3 = st.tabs(["📷 Original", "🔬 Enhanced", "🌡️ Grad-CAM"])

            with tab1:
                # FIX: use_column_width for newer Streamlit versions
                st.image(Image.open(tmp_path), caption="Original X-ray",
                         use_column_width=True)

            with tab2:
                enhanced = apply_clahe_display(tmp_path)
                st.image(enhanced, caption="CLAHE Enhanced (OpenCV)",
                         use_column_width=True)

            with tab3:
                cam_overlay = result.get("gradcam_overlay")
                if cam_overlay is not None:
                    st.image(cam_overlay,
                             caption="Grad-CAM Activation — Model focus regions",
                             use_column_width=True)
                    st.info("🔴 Red/yellow regions = areas the model focused on most")

        with chart_col:
            fig_prob = make_probability_chart(result["class_probs"])
            st.plotly_chart(fig_prob, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig_gauge = make_severity_gauge(conf, result["severity"])
                st.plotly_chart(fig_gauge, use_container_width=True)
            with c2:
                fig_radar = make_metrics_radar(result["cv_metrics"])
                st.plotly_chart(fig_radar, use_container_width=True)

        # ── OpenCV metrics ────────────────────────────────────────────────────
        st.markdown("#### 🔬 Image Quality Metrics (OpenCV Pipeline)")
        m = result["cv_metrics"]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        metrics_display = [
            (mc1, f"{m.get('mean_intensity',0):.1f}", "Mean Intensity"),
            (mc2, f"{m.get('std_intensity',0):.1f}",  "Std Intensity"),
            (mc3, f"{m.get('contrast_ratio',0):.3f}", "Contrast Ratio"),
            (mc4, f"{m.get('edge_density',0):.3f}",   "Edge Density"),
            (mc5, f"{m.get('lung_area_ratio',0)*100:.1f}%", "Lung Area"),
        ]
        for col, val, label in metrics_display:
            col.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── RAG Explanation ───────────────────────────────────────────────────
        st.markdown("### 🗂️ RAG-Powered Clinical Explanation")
        st.caption("Retrieved from medical knowledge base · Grounded in WHO/ATS guidelines")

        if rag_result.get("sources"):
            st.markdown("**📚 Knowledge Sources Retrieved:**")
            source_html = " ".join(
                f'<span class="source-chip">📄 {s["title"]}</span>'
                for s in rag_result["sources"]
            )
            st.markdown(source_html, unsafe_allow_html=True)
            st.markdown("")

        with st.expander("📋 Full Clinical Explanation (click to expand)", expanded=True):
            st.markdown(rag_result["explanation"])

        # ── Follow-up Q&A ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 💬 Ask a Clinical Question")
        st.caption("Powered by RAG — answers grounded in medical literature")

        followup_q = st.text_input(
            "Your question:",
            placeholder="e.g. What antibiotic should be prescribed? Is hospitalization needed?",
        )

        st.markdown("**Quick questions:**")
        q_cols = st.columns(3)
        follow_ups = rag_result.get("followup_questions", [])[:3]
        for i, (qcol, fq) in enumerate(zip(q_cols, follow_ups)):
            if qcol.button(f"❓ {fq[:45]}…" if len(fq) > 45 else f"❓ {fq}",
                           key=f"fq_{i}"):
                followup_q = fq

        if followup_q:
            with st.spinner("🤔 Retrieving answer from medical knowledge base..."):
                rag = load_rag(openai_key)
                answer = rag.generate_explanation(
                    predicted_class=result["predicted_class"],
                    confidence=result["confidence"],
                    cv_metrics=result["cv_metrics"],
                    class_probs=result["class_probs"],
                    custom_question=followup_q,
                )
            st.markdown("#### 💡 Answer:")
            st.markdown(answer["explanation"])

        # ── PDF Report download ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📄 Download Report")

        col_pdf, col_json = st.columns(2)
        with col_pdf:
            if st.button("📄 Generate PDF Report", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf_path = os.path.join(tempfile.gettempdir(), "medical_report.pdf")
                    out = generate_pdf_report(
                        result=result,
                        explanation=rag_result["explanation"],
                        image_path=tmp_path,
                        output_path=pdf_path,
                    )
                    with open(out, "rb") as f:
                        st.download_button(
                            "⬇️ Download PDF Report",
                            data=f.read(),
                            file_name="ai_medical_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

        with col_json:
            import json
            report_data = {
                "predicted_class": result["predicted_class"],
                "confidence":      result["confidence"],
                "severity":        result["severity"],
                "class_probs":     result["class_probs"],
                "cv_metrics":      result["cv_metrics"],
                "explanation":     rag_result["explanation"],
                "sources":         [s["title"] for s in rag_result.get("sources", [])],
            }
            st.download_button(
                "⬇️ Download JSON Results",
                data=json.dumps(report_data, indent=2),
                file_name="ai_medical_results.json",
                mime="application/json",
                use_container_width=True,
            )

    else:
        # Show preview while not analyzed
        st.markdown("#### Preview")
        preview_col, info_col = st.columns([1, 2])
        with preview_col:
            # FIX: use_column_width for newer Streamlit versions
            st.image(Image.open(uploaded_file), caption="Uploaded image",
                     use_column_width=True)
        with info_col:
            st.info(
                "✅ Image uploaded successfully!\n\n"
                "Click **Analyze X-ray** to run:\n"
                "1. OpenCV preprocessing (CLAHE, denoising)\n"
                "2. ResNet-50 disease classification\n"
                "3. Grad-CAM visualization\n"
                "4. FAISS RAG knowledge retrieval\n"
                "5. GPT-4 clinical explanation"
            )
