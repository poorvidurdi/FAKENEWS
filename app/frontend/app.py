import streamlit as st
import requests

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Fake News Detection System",
    layout="centered"
)

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Text Fake News Detection", "Multimodal Fake News Detection"]
)

# ==================================================
# PAGE 1 — TEXT FAKE NEWS DETECTION (UNCHANGED)
# ==================================================
if page == "Text Fake News Detection":

    st.title("📰 Text Fake News Detection")
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
                st.markdown("⚠️ **Suspicious words influencing the decision**")
                st.write(", ".join(suspicious_words))


# ==================================================
# PAGE 2 — MULTIMODAL FAKE NEWS DETECTION
# ==================================================
else:

    st.title("🧠 Multimodal Fake News Detection")
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

            st.subheader("🔍 Multimodal Analysis Result")

            st.write("**Text Prediction:**", result["text_prediction"])
            st.write("**Text Confidence:**", result["text_confidence"])

            st.write("**Image Out-of-Context:**", result["image_out_of_context"])
            st.write("**Multimodal Score:**", result["multimodal_score"])

            final = result["final_decision"]

            if final == "REAL":
                st.success("FINAL DECISION: REAL NEWS")
            else:
                st.error("FINAL DECISION: FAKE NEWS")
                
                fake_reasons = result.get("fake_reasons", [])
                if fake_reasons:
                    st.write("**Reason(s):** " + " AND ".join(fake_reasons))
                
            suspicious_words = result.get("suspicious_words", [])
            if suspicious_words:
                st.markdown("⚠️ **Suspicious words influencing the decision**")
                st.write(", ".join(suspicious_words))
