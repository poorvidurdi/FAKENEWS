import streamlit as st
import requests
import json
import os

# --------------------------------------------------
# CONFIG & STYLE
# --------------------------------------------------
st.set_page_config(
    page_title="Forensics AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    local_css(css_path)

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

def go_to_app():
    st.session_state.page = 'app'

# --------------------------------------------------
# LANDING PAGE
# --------------------------------------------------
<<<<<<< HEAD
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
=======
def show_landing_page():
    # Hero Section
    st.markdown('<div style="height: 10vh;"></div>', unsafe_allow_html=True)
>>>>>>> 4420803 (Update frontend UI with assets and styling for AI forensics system)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="landing-title" style="text-align: center;">AI Forensics System</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="landing-subtitle" style="text-align: center;">'
            'Advanced multimodal system for detecting misinformation using linguistic analysis '
            'and image forensics.'
            '</p>', 
            unsafe_allow_html=True
        )
        
        # Centered Start Button
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Proceed", width='stretch'):
                go_to_app()
                st.rerun()


    st.markdown('<div style="height: 5vh;"></div>', unsafe_allow_html=True)

    # How it Works Section
    st.markdown('<h3 style="text-align: center; margin-bottom: 2rem;">System Capabilities</h3>', unsafe_allow_html=True)
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">Text Analysis</div>
            <div class="card-text">Neural NLP engine for linguistic forensics and pattern detection.</div>
            <div class="card-detail">
                <div class="detail-item"><div class="detail-dot"></div> TF-IDF Semantic Analysis</div>
                <div class="detail-item"><div class="detail-dot"></div> Sentiment & Tone Forensic</div>
                <div class="detail-item"><div class="detail-dot"></div> Official Source Verification</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">Image Forensics</div>
            <div class="card-text">Computer vision suite for pixel-level manipulation discovery.</div>
            <div class="card-detail">
                <div class="detail-item"><div class="detail-dot"></div> ResNet-18 Feature Extraction</div>
                <div class="detail-item"><div class="detail-dot"></div> Grad-CAM Heatmap Analysis</div>
                <div class="detail-item"><div class="detail-dot"></div> Pixel Inconsistency Detect</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_c:
        st.markdown("""
        <div class="info-card">
            <div class="card-title">Multimodal Fusion</div>
            <div class="card-text">Cross-modal analysis correlating claims with visual evidence.</div>
            <div class="card-detail">
                <div class="detail-item"><div class="detail-dot"></div> Multimodal Logic Sync</div>
                <div class="detail-item"><div class="detail-dot"></div> Contextual Consistency Check</div>
                <div class="detail-item"><div class="detail-dot"></div> Joint Credibility Scoring</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------
def show_main_app():
    # Sidebar Logo and Premium Header
    st.sidebar.markdown("""
    <div class="sidebar-logo-container">
        <div class="sidebar-logo-text">FORENSICS AI</div>
        <div class="sidebar-tagline">Advanced Truth Verification</div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.radio(
        "SELECT MODULE",
        ["Text Analysis", "Multimodal Analysis", "Image Forensics", "System Metrics"]
    )

    # PAGE 1: TEXT ANALYSIS
    if page == "Text Analysis":
        st.title("Text Analysis")
        st.write("Analyze news articles for linguistic indicators of misinformation.")
        
        news_text = st.text_area("Input Text", height=200)
        
        if st.button("Analyze Text"):
            with st.spinner("Processing..."):
                try:
                    response = requests.post("http://127.0.0.1:5000/predict-text", json={"text": news_text})
                    if response.status_code == 200:
                        result = response.json()
                        label = result["label"]
                        score = result["probability_score"]
                        
                        # Determine Style based on Label
                        if label == "FAKE":
                            card_class = "result-card-fake"
                            text_color = "#ef4444" # Red
                        elif label == "REAL":
                            card_class = "result-card-real"
                            text_color = "#22c55e" # Green
                        elif label == "SUSPICIOUS":
                            card_class = "result-card-uncertain"
                            text_color = "#f97316" # Orange
                        else:
                            card_class = "result-card-uncertain"
                            text_color = "#eab308" # Amber
                            
                        # Custom Result Card
                        st.markdown(f"""
                        <div class="{card_class}">
                            <div class="result-large-text" style="color: {text_color}; text-shadow: 0 0 10px {text_color}44;">{label}</div>
                            <div class="result-sub-text">Probability score: <strong>{score:.3f}</strong></div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Natural Language Explanation
                        st.markdown(f'<div style="border-left: 4px solid {text_color}; padding-left: 1rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
                        st.markdown(result.get("explanation", "No explanation available."))
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Highlight Text Logic (Using RegEx for word boundaries)
                        import re
                        highlighted_text = news_text
                        if result.get("suspicious_words"):
                            for word in result["suspicious_words"]:
                                # Regex to match whole word, case insensitive
                                pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
                                highlighted_text = pattern.sub(r'<span class="highlight-sus">\1</span>', highlighted_text)
                        
                        st.markdown('<h4 style="margin-top: 1rem;">Evidence Detail:</h4>', unsafe_allow_html=True)
                        st.caption("Highlighted words contributing to the classification (neutral terms excluded):")
                        st.markdown(
                            f'<div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 8px; line-height: 1.8; font-size: 1.05rem;">{highlighted_text}</div>', 
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    st.error(f"Connection error: {e}")

    # PAGE 2: MULTIMODAL ANALYSIS
    elif page == "Multimodal Analysis":
        st.title("Multimodal Analysis")
        st.write("Correlate text claims with image evidence.")
        
        text_input = st.text_area("Claim / Text", height=150)
        img_input = st.file_uploader("Evidence Image", type=["jpg", "png"])
        
        if st.button("Run Fusion Analysis"):
            if text_input and img_input:
                with st.spinner("Correlating signals..."):
                    try:
                        files = {"image": img_input}
                        data = {"text": text_input}
                        response = requests.post("http://127.0.0.1:5002/predict_multimodal", files=files, data=data)
                        
                        if response.status_code == 200:
                            res = response.json()
                            
                            # Standardized UI
                            category = res["category"] # Fake/Real
                            score = res["probability_score"]
                            explanation = res["explanation"]
                            
                            if category == "Fake":
                                card_class = "result-card-fake"
                                text_color = "#ef4444"
                            else:
                                card_class = "result-card-real"
                                text_color = "#22c55e"
                                
                            st.markdown(f"""
                            <div class="{card_class}">
                                <div class="result-large-text" style="color: {text_color};">{category.upper()}</div>
                                <div class="result-sub-text">Probability score: <strong>{score}</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style="background: rgba(255, 255, 255, 0.05); border-left: 4px solid {text_color}; padding: 1rem; border-radius: 4px; margin-bottom: 2rem;">
                                <strong>Fusion Analysis:</strong> {explanation}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Details
                            if res["details"].get("fake_reasons"):
                                st.write("Risk Factors:")
                                for reason in res["details"]["fake_reasons"]:
                                    st.write(f"- {reason}")
                    except Exception as e:
                         st.error(str(e))
                    except:
                        st.error("Backend unavailable.")

    # PAGE 3: IMAGE FORENSICS
    elif page == "Image Forensics":
        st.title("Image Forensics")
        st.write("Scan images for digital manipulation traces.")
        
        img_file = st.file_uploader("Upload Image", type=["jpg", "png"])
        
        if st.button("Scan Image"):
            if img_file:
                with st.spinner("Scanning pixel artifacts..."):
                    try:
                        files = {"image": img_file}
                        response = requests.post("http://127.0.0.1:5001/predict-image", files=files)
                        
                        if response.status_code == 200:
                            res = response.json()
                            st.subheader("Forensic Report")
                            
                            # High-Visibility UI (Keeping the "font thing")
                            label = res.get("label", "Unknown")
                            confidence = res.get("confidence", 0.0)
                            
                            if label == "FAKE":
                                card_class = "result-card-fake"
                                text_color = "#ef4444" 
                            else:
                                card_class = "result-card-real"
                                text_color = "#22c55e"
                                
                            st.markdown(f"""
                            <div class="{card_class}">
                                <div class="result-large-text" style="color: {text_color}; text-shadow: 0 0 10px {text_color}44;">{label}</div>
                                <div class="result-sub-text">Confidence: <strong>{confidence:.3f}</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if res.get("reasons"):
                                st.write("Analysis Details:")
                                for reason in res["reasons"]:
                                    st.write(f"- {reason}")
                            
                            if res.get("heatmap_path") and os.path.exists(res.get("heatmap_path")):
                                st.image(res["heatmap_path"], caption="Manipulation Heatmap", width='stretch')
                    except Exception as e:
                        st.error(f"Backend error: {e}")

    # PAGE 4: METRICS
    else:
        st.title("System Performance Metrics")
        st.write("Comprehensive evaluation of AI models across text, image, and multimodal domains.")
        
        metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results", "metrics.json")
        
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
            
            # 1. Overall System Summary
            st.markdown('<div class="module-header">Global Performance Overview</div>', unsafe_allow_html=True)
            o = data.get("system_overall", {})
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="metric-tile"><div class="metric-value">{o.get("accuracy", 0):.1%}</div><div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="metric-tile"><div class="metric-value">{o.get("precision", 0):.1%}</div><div class="metric-label">Precision</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="metric-tile"><div class="metric-value">{o.get("recall", 0):.1%}</div><div class="metric-label">Recall</div></div>', unsafe_allow_html=True)
            with c4:
                st.markdown(f'<div class="metric-tile" style="border-color: #818cf8;"><div class="metric-value" style="color: #818cf8;">{o.get("f1_score", 0):.1%}</div><div class="metric-label">F1 Score</div></div>', unsafe_allow_html=True)
            
            # 2. Model Comparison Chart
            st.markdown('<div class="module-header">Cross-Module Benchmarking</div>', unsafe_allow_html=True)
            
            chart_data = {
                "Model": ["Text Model", "Image Model", "Multimodal Model"],
                "Accuracy": [data["text_model"]["accuracy"], data["image_model"]["accuracy"], data["multimodal_model"]["accuracy"]],
                "F1 Score": [data["text_model"]["f1_score"], data["image_model"]["f1_score"], data["multimodal_model"]["f1_score"]]
            }
            st.bar_chart(chart_data, x="Model", y=["Accuracy", "F1 Score"])

            # 3. Model Breakdown
            st.markdown('<div class="module-header">Detailed Model Breakdown</div>', unsafe_allow_html=True)
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.subheader("Text Engine")
                st.write(f"Precision: {data['text_model']['precision']:.3f}")
                st.write(f"Recall: {data['text_model']['recall']:.3f}")
                st.write(f"F1: {data['text_model']['f1_score']:.3f}")
            with m2:
                st.subheader("Visual Engine")
                st.write(f"Precision: {data['image_model']['precision']:.3f}")
                st.write(f"Recall: {data['image_model']['recall']:.3f}")
                st.write(f"F1: {data['image_model']['f1_score']:.3f}")
            with m3:
                st.subheader("Fusion Engine")
                st.write(f"Precision: {data['multimodal_model']['precision']:.3f}")
                st.write(f"Recall: {data['multimodal_model']['recall']:.3f}")
                st.write(f"F1: {data['multimodal_model']['f1_score']:.3f}")

        else:
            st.warning("No metrics data available. Please run model evaluation.")

# --------------------------------------------------
# ROUTER
# --------------------------------------------------
if st.session_state.page == 'landing':
    show_landing_page()
else:
    show_main_app()
