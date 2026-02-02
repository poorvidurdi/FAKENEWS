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

def show_landing_page():
    # Hero Section
    st.markdown('<div style="height: 10vh;"></div>', unsafe_allow_html=True)
    
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
        
        # Centered Proceed Button
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Proceed", use_container_width=True):
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
                    st.error(f"Text Backend Error: {e}")
                    st.info("Ensure the Text API is running on port 5000.")

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
                         st.error(f"Multimodal Backend Error: {e}")
                         st.info("Ensure the Multimodal API is running on port 5002.")

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
                        st.error(f"Image Forensics Backend Error: {e}")
                        st.info("Ensure the Image API is running on port 5001.")

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
            
            # 2. Model Breakdown
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
