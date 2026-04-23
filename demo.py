"""
demo.py
Quick end-to-end demo that runs without a UI.
Creates a synthetic test image, runs the full pipeline, and prints results.

Usage:
  python demo.py
  python demo.py --image path/to/xray.jpg
"""

import argparse
import os
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([
    os.path.join(ROOT, "models"),
    os.path.join(ROOT, "rag"),
    os.path.join(ROOT, "utils"),
])


def create_synthetic_xray(path: str):
    """Create a synthetic grayscale image resembling an X-ray for testing."""
    img = np.zeros((512, 512), dtype=np.uint8)
    # Background lung field
    cv2.ellipse(img, (160, 280), (120, 180), 0, 0, 360, 80, -1)
    cv2.ellipse(img, (350, 280), (120, 180), 0, 0, 360, 80, -1)
    # Heart silhouette
    cv2.ellipse(img, (245, 320), (70, 90), 0, 0, 360, 40, -1)
    # Ribs
    for i in range(6):
        y = 160 + i * 40
        cv2.ellipse(img, (256, y), (200 - i*5, 20), 0, 0, 180, 130, 2)
    # Add noise
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    print(f"[Demo] Synthetic X-ray saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="AI Medical Image Analyzer - Demo")
    parser.add_argument("--image",      default=None,  help="Path to chest X-ray image")
    parser.add_argument("--model_path", default=None,  help="Path to trained model weights")
    parser.add_argument("--openai_key", default=None,  help="OpenAI API key")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))

    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY", "")

    # ── 1. Prepare test image ─────────────────────────────────────────────────
    if args.image and os.path.exists(args.image):
        image_path = args.image
        print(f"[Demo] Using provided image: {image_path}")
    else:
        os.makedirs("data", exist_ok=True)
        image_path = "data/test_xray.png"
        create_synthetic_xray(image_path)

    # ── 2. Run classification ─────────────────────────────────────────────────
    print("\n[Demo] ── Step 1-3: OpenCV + ResNet-50 inference ──")
    from classifier import MedicalImageAnalyzer
    model_path = args.model_path or os.getenv("MODEL_PATH", "models/chest_xray_resnet50.pth")
    analyzer   = MedicalImageAnalyzer(model_path=model_path)
    result     = analyzer.analyze(image_path)

    print(f"\n{'='*50}")
    print(f"  Predicted Class : {result['predicted_class']}")
    print(f"  Confidence      : {result['confidence']*100:.1f}%")
    print(f"  Severity        : {result['severity']}")
    print(f"\n  Class Probabilities:")
    for cls, p in sorted(result['class_probs'].items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 40)
        print(f"    {cls:<20} {p*100:5.1f}%  {bar}")
    print(f"\n  OpenCV Metrics:")
    for k, v in result['cv_metrics'].items():
        print(f"    {k:<22} {v:.4f}")
    print(f"{'='*50}\n")

    # ── 3. RAG explanation ────────────────────────────────────────────────────
    print("[Demo] ── Step 4-5: RAG retrieval + AI explanation ──")
    from rag_engine import MedicalRAGEngine
    rag    = MedicalRAGEngine(openai_api_key=openai_key)
    answer = rag.generate_explanation(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        cv_metrics=result["cv_metrics"],
        class_probs=result["class_probs"],
    )

    print("\n📚 Retrieved Sources:")
    for s in answer["sources"]:
        print(f"  • [{s['score']:.3f}] {s['title']}")

    print("\n💬 AI Clinical Explanation:")
    print("-" * 50)
    print(answer["explanation"][:1500])
    if len(answer["explanation"]) > 1500:
        print("... [truncated — full explanation in Streamlit UI]")
    print("-" * 50)

    print("\n❓ Suggested Follow-up Questions:")
    for q in answer["followup_questions"]:
        print(f"  → {q}")

    # ── 4. Grad-CAM image ─────────────────────────────────────────────────────
    cam_path = "data/gradcam_output.png"
    if result.get("gradcam_overlay") is not None:
        Image.fromarray(result["gradcam_overlay"]).save(cam_path)
        print(f"\n[Demo] Grad-CAM saved to: {cam_path}")

    print("\n✅ Demo complete!")
    print("   Run the Streamlit UI with:  streamlit run app.py")


if __name__ == "__main__":
    main()
