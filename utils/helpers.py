"""
utils/helpers.py
Utility functions: PDF report generation, chart helpers, image utils.
"""

import io
import os
import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import plotly.graph_objects as go
import plotly.express as px


# ── Plotly charts ─────────────────────────────────────────────────────────────
def make_probability_chart(class_probs: dict) -> go.Figure:
    """Horizontal bar chart of class probabilities."""
    classes = list(class_probs.keys())
    probs   = [v * 100 for v in class_probs.values()]
    colors  = []
    max_p   = max(probs)

    color_map = {
        "Normal":           "#27AE60",
        "Pneumonia":        "#E67E22",
        "COVID-19":         "#E74C3C",
        "Tuberculosis":     "#8E44AD",
        "Pleural Effusion": "#2980B9",
    }
    for cls, p in zip(classes, probs):
        colors.append(color_map.get(cls, "#95A5A6"))

    fig = go.Figure(go.Bar(
        x=probs,
        y=classes,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Diagnosis Probability Distribution",
        xaxis_title="Probability (%)",
        xaxis=dict(range=[0, 115]),
        height=280,
        margin=dict(l=10, r=60, t=40, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    return fig


def make_metrics_radar(cv_metrics: dict) -> go.Figure:
    """Radar chart of image quality metrics."""
    labels  = ["Mean Intensity", "Std Intensity", "Contrast Ratio",
                "Edge Density", "Lung Area"]
    values  = [
        min(cv_metrics.get("mean_intensity",  0) / 255, 1),
        min(cv_metrics.get("std_intensity",   0) / 128, 1),
        min(cv_metrics.get("contrast_ratio",  0),       1),
        min(cv_metrics.get("edge_density",    0) * 10,  1),
        min(cv_metrics.get("lung_area_ratio", 0),       1),
    ]
    values_pct = [v * 100 for v in values]

    fig = go.Figure(go.Scatterpolar(
        r=values_pct + [values_pct[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(41, 128, 185, 0.2)",
        line=dict(color="#2980B9", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Image Quality Metrics",
        height=300,
        margin=dict(l=40, r=40, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_severity_gauge(confidence: float, severity: str) -> go.Figure:
    """Gauge chart showing confidence and severity."""
    severity_color = {
        "none":     "#27AE60",
        "moderate": "#E67E22",
        "high":     "#E74C3C",
    }.get(severity, "#95A5A6")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": severity_color},
            "steps": [
                {"range": [0,  50], "color": "#FADBD8"},
                {"range": [50, 75], "color": "#FDEBD0"},
                {"range": [75, 100],"color": "#D5F5E3"},
            ],
            "threshold": {
                "line":  {"color": "#2C3E50", "width": 3},
                "thickness": 0.85,
                "value": confidence * 100,
            },
        },
        title={"text": "Model Confidence"},
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ── Image utilities ───────────────────────────────────────────────────────────
def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    return np.array(pil_img)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_temp_image(pil_or_array, path: str) -> str:
    """Save PIL image or numpy array to disk; return path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(pil_or_array, np.ndarray):
        pil_or_array = numpy_to_pil(pil_or_array)
    pil_or_array.save(path)
    return path


def apply_clahe_display(image_path: str) -> np.ndarray:
    """Return CLAHE-enhanced RGB image for display."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((224, 224, 3), dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


# ── Severity badge ────────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "none":     ("#D5F5E3", "#1E8449", "✅ Normal — No pathology detected"),
    "moderate": ("#FDEBD0", "#CA6F1E", "⚠️ Moderate — Clinical attention advised"),
    "high":     ("#FADBD8", "#C0392B", "🚨 High — Urgent medical evaluation needed"),
}


def get_severity_info(severity: str) -> tuple[str, str, str]:
    return SEVERITY_COLORS.get(severity, ("#F2F3F4", "#566573", "Unknown"))


# ── PDF Report ────────────────────────────────────────────────────────────────
def generate_pdf_report(result: dict, explanation: str,
                         image_path: str, output_path: str) -> str:
    """
    Generate a clinical PDF report using ReportLab.
    Returns the output path.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Image as RLImage, Table, TableStyle,
                                         HRFlowable)

        doc    = SimpleDocTemplate(output_path, pagesize=A4,
                                   topMargin=2*cm, bottomMargin=2*cm,
                                   leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # Title
        title_style = ParagraphStyle(
            "Title", parent=styles["Title"],
            fontSize=18, spaceAfter=6,
            textColor=colors.HexColor("#1A5276"),
        )
        story.append(Paragraph("AI Medical Image Analysis Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | "
            f"System: AI Medical Analyzer v1.0",
            styles["Normal"]
        ))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#1A5276")))
        story.append(Spacer(1, 0.4*cm))

        # Diagnosis summary table
        pred  = result["predicted_class"]
        conf  = result["confidence"]
        sev   = result["severity"]
        story.append(Paragraph("Diagnosis Summary", styles["Heading2"]))

        sev_color = {"none": "#27AE60", "moderate": "#E67E22",
                     "high": "#E74C3C"}.get(sev, "#95A5A6")
        data = [
            ["Parameter",          "Value"],
            ["Predicted Diagnosis", pred],
            ["Model Confidence",    f"{conf*100:.1f}%"],
            ["Severity",            sev.capitalize()],
        ]
        t = Table(data, colWidths=[7*cm, 10*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#1A5276")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 11),
            ("BACKGROUND",  (0, 1), (-1, -1), colors.HexColor("#EBF5FB")),
            ("ROWBACKGROUNDS", (0, 2), (-1, -1),
             [colors.white, colors.HexColor("#EBF5FB")]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING",     (0, 0), (-1, -1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # Class probabilities
        story.append(Paragraph("Probability Distribution", styles["Heading2"]))
        prob_data = [["Condition", "Probability"]] + [
            [cls, f"{p*100:.1f}%"]
            for cls, p in sorted(result["class_probs"].items(),
                                 key=lambda x: -x[1])
        ]
        pt = Table(prob_data, colWidths=[10*cm, 7*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A5276")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING",     (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#EBF5FB")]),
        ]))
        story.append(pt)
        story.append(Spacer(1, 0.5*cm))

        # AI explanation (plain text, stripped of markdown)
        story.append(Paragraph("AI Clinical Explanation", styles["Heading2"]))
        clean = explanation.replace("**", "").replace("##", "").replace("*", "")
        for line in clean.split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 0.15*cm))

        # Disclaimer
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        disclaimer_style = ParagraphStyle(
            "Disclaimer", parent=styles["Normal"],
            fontSize=9, textColor=colors.grey,
        )
        story.append(Paragraph(
            "DISCLAIMER: This report is generated by an AI system for decision support only. "
            "It does not constitute a medical diagnosis. All findings must be interpreted by "
            "a qualified radiologist or physician in the context of the patient's full clinical "
            "history. Do not use this report as the sole basis for clinical decisions.",
            disclaimer_style,
        ))

        doc.build(story)
        return output_path

    except ImportError:
        # Fallback: plain text report
        txt_path = output_path.replace(".pdf", ".txt")
        with open(txt_path, "w") as f:
            f.write(f"AI Medical Image Analysis Report\n")
            f.write(f"Generated: {datetime.datetime.now()}\n\n")
            f.write(f"Diagnosis: {result['predicted_class']}\n")
            f.write(f"Confidence: {result['confidence']*100:.1f}%\n\n")
            f.write(explanation)
        return txt_path
