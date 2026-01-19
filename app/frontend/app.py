import streamlit as st
import requests
import json
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection System",
    layout="wide"
)

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Text Fake News Detection", "Multimodal Fake News Detection", "Image Fake News Detection", "Model Analysis"]
)

# ==================================================
# PAGE 1 — TEXT FAKE NEWS DETECTION (UNCHANGED)
# ==================================================
if page == "Text Fake News Detection":

    st.title("Text Fake News Detection")
    st.write(
        "Enter a news statement below. The model will classify it as "
        "**REAL**, **FAKE**, or **UNCERTAIN**, and highlight suspicious terms "
        "that influenced the decision."
    )

    news_text = st.text_area(
        "News Text",
        placeholder="Paste or type the news text here...",
        height=180
    )

    if st.button("Check"):
        if not news_text.strip():
            st.warning("Please enter some text before checking.")
        else:
            with st.spinner("Analyzing text..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:5000/predict-text",
                        json={"text": news_text},
                        timeout=10
                    )
                except requests.exceptions.RequestException:
                    st.error("Backend API is not running. Please start the Flask server.")
                    st.stop()

            if response.status_code != 200:
                st.error("Error received from backend API.")
                st.stop()

            result = response.json()

            label = result["label"]
            probability = result["fake_probability"]
            suspicious_words = result["suspicious_words"]

            if label == "REAL":
                st.success("REAL NEWS")
                st.write(
                    f"**Fake Probability:** {probability}\n\n"
                    "The text does not show strong linguistic patterns of misinformation."
                )

            elif label == "UNCERTAIN":
                st.warning("UNCERTAIN")
                st.write(
                    f"**Fake Probability:** {probability}\n\n"
                    "The text contains potentially misleading language and should be verified."
                )

            else:
                st.error("FAKE NEWS")
                st.write(
                    f"**Fake Probability:** {probability}\n\n"
                    "The text shows strong indicators of misinformation."
                )

            if suspicious_words:
                st.markdown("**Suspicious words influencing the decision**")
                st.write(", ".join(suspicious_words))


# ==================================================
# PAGE 2 — MULTIMODAL FAKE NEWS DETECTION
# ==================================================
elif page == "Multimodal Fake News Detection":

    st.title("Multimodal Fake News Detection")
    st.write(
        "Analyze **text + image together** to detect **REAL**, **FAKE**, "
        "or **OUT-OF-CONTEXT** news."
    )

    multimodal_text = st.text_area(
        "News Text",
        placeholder="Paste or type the news text here...",
        height=160
    )

    multimodal_image = st.file_uploader(
        "Upload Related Image",
        type=["jpg", "jpeg", "png"]
    )

    if st.button("Analyze Multimodal Content"):
        if not multimodal_text.strip() or multimodal_image is None:
            st.warning("Please provide both text and image.")
        else:
            with st.spinner("Analyzing multimodal content..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:5002/predict_multimodal",
                        files={"image": multimodal_image},
                        data={"text": multimodal_text},
                        timeout=15
                    )
                except requests.exceptions.RequestException:
                    st.error("Multimodal backend is not running.")
                    st.stop()

            if response.status_code != 200:
                st.error("Error received from multimodal backend.")
                st.stop()

            result = response.json()

            st.subheader("Multimodal Analysis Result")

            st.write("**Text Prediction:**", result["text_prediction"])
            st.write("**Text Confidence:**", result["text_confidence"])

            st.write("**Image Out-of-Context:**", result["image_out_of_context"])
            st.write("**Multimodal Score:**", result["multimodal_score"])

            final = result["final_decision"]

            if final == "REAL":
                st.success("FINAL DECISION: REAL NEWS")
            elif "UNCERTAIN" in final:
                st.warning(f"FINAL DECISION: {final}")
                fake_reasons = result.get("fake_reasons", [])
                if fake_reasons:
                    st.info("**Note:** " + " AND ".join(fake_reasons))
            else:
                st.error(f"FINAL DECISION: {final}")
                
                fake_reasons = result.get("fake_reasons", [])
                if fake_reasons:
                    st.write("**Reason(s):** " + " AND ".join(fake_reasons))
                
            suspicious_words = result.get("suspicious_words", [])
            if suspicious_words:
                st.markdown("**Suspicious words influencing the decision**")
                st.write(", ".join(suspicious_words))

# ==================================================
# PAGE 3 — IMAGE FAKE NEWS DETECTION
# ==================================================
elif page == "Image Fake News Detection":

    st.title("Image Fake News Detection")
    st.write(
        "Upload an image to detect if it is **REAL** or **FAKE** (manipulated). "
        "For fake images, we provide a **heatmap** showing manipulated regions."
    )

    image_file = st.file_uploader(
        "Upload Image for Analysis",
        type=["jpg", "jpeg", "png"]
    )

    if st.button("Analyze Image"):
        if image_file is None:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Analyzing image forensics..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:5001/predict-image",
                        files={"image": image_file},
                        timeout=20
                    )
                except requests.exceptions.RequestException:
                    st.error("Image backend is not running. Please start the image API.")
                    st.stop()

            if response.status_code != 200:
                st.error("Error received from image backend.")
                st.stop()

            result = response.json()

            st.subheader("Image Analysis Result")
            
            label = result["label"]
            conf = result["confidence"]
            reasons = result["reasons"]
            heatmap_path = result["heatmap_path"]

            if label == "REAL":
                st.success(f"DECISION: REAL IMAGE (Confidence: {conf})")
                st.write(reasons[0])
            else:
                st.error(f"DECISION: FAKE / MANIPULATED (Confidence: {conf})")
                st.write("**Reason(s):**")
                for r in reasons:
                    st.write(f"- {r}")
                
                if heatmap_path and os.path.exists(heatmap_path):
                    st.subheader("Manipulation Heatmap")
                    st.image(heatmap_path, caption="Heatmap: Red areas indicate higher probability of manipulation.")
                    
                    # Add heat map color gradient legend
                    st.markdown("""
                        <div style="display: flex; align-items: center; gap: 10px; margin-top: 10px;">
                            <span style="font-size: 0.9rem; color: #555;">Real (Low)</span>
                            <div style="flex-grow: 1; height: 15px; background: linear_gradient(to right, blue, cyan, green, yellow, red); border-radius: 3px;"></div>
                            <span style="font-size: 0.9rem; color: #555;">Fake (High)</span>
                        </div>
                        <style>
                            .heatmap-legend {
                                background: linear-gradient(to right, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000);
                                height: 12px;
                                width: 100%;
                                border-radius: 5px;
                                margin: 5px 0;
                            }
                        </style>
                        <div class="heatmap-legend"></div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("Heatmap generation failed or not available.")

# ==================================================
# PAGE 4 — MODEL ANALYSIS
# ==================================================
else:
    st.title("📊 Model Performance Analysis")
    st.write("Performance metrics for both Text and Multimodal models based on the latest evaluations.")

    metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "metrics.json")
    
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Text Model")
            text_metrics = metrics.get("text_model", {})
            if text_metrics:
                st.metric("Accuracy", f"{text_metrics['accuracy']:.2%}")
                st.metric("F1-Score", f"{text_metrics['f1_score']:.2f}")
            else:
                st.info("Text metrics unavailable.")

        with col2:
            st.subheader("Multimodal")
            mm_metrics = metrics.get("multimodal_model", {})
            if mm_metrics:
                st.metric("Accuracy", f"{mm_metrics['accuracy']:.2%}")
                st.metric("F1-Score", f"{mm_metrics['f1_score']:.2f}")
            else:
                st.info("Multimodal metrics unavailable.")

        with col3:
            st.subheader("Image Model")
            image_metrics = metrics.get("image_model", {})
            if image_metrics:
                st.metric("Accuracy", f"{image_metrics['accuracy']:.2%}")
                st.metric("F1-Score", f"{image_metrics['f1_score']:.2f}")
            else:
                st.info("Image metrics unavailable.")
    else:
        st.error("Metrics data file not found. Please run the metrics generation script.")

