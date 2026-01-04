import streamlit as st
import requests

# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Text Fake News Detector",
    layout="centered"
)

# --------------------------------------------------
# UI Header
# --------------------------------------------------

st.title("📰 Text Fake News Detection")
st.write(
    "Enter a news statement below. The model will classify it as "
    "**REAL**, **FAKE**, or **UNCERTAIN**, and highlight suspicious terms "
    "that influenced the decision."
)

# --------------------------------------------------
# Text input
# --------------------------------------------------

news_text = st.text_area(
    "News Text",
    placeholder="Paste or type the news text here...",
    height=180
)

# --------------------------------------------------
# Prediction button
# --------------------------------------------------

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

        # --------------------------------------------------
        # Display result
        # --------------------------------------------------

        if label == "REAL":
            st.success(f"🟢 REAL NEWS")
            st.write(
                f"**Fake Probability:** {probability}\n\n"
                "The text does not show strong linguistic patterns of misinformation."
            )

        elif label == "UNCERTAIN":
            st.warning(f"🟡 UNCERTAIN")
            st.write(
                f"**Fake Probability:** {probability}\n\n"
                "The text contains potentially misleading language and should be verified."
            )

        else:
            st.error(f"🔴 FAKE NEWS")
            st.write(
                f"**Fake Probability:** {probability}\n\n"
                "The text shows strong indicators of misinformation."
            )

        # --------------------------------------------------
        # Suspicious words
        # --------------------------------------------------

        if suspicious_words:
            st.markdown("⚠️ Suspicious words influencing the decision")
            st.write(", ".join(suspicious_words))