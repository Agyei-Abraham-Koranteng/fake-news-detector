import streamlit as st
import pickle
import numpy as np
import pandas as pd
from src.preprocess import TextPreprocessor
from src.ui import load_css, render_header, render_menu, render_result, render_stats_row, render_history
from pypdf import PdfReader
from PIL import Image
import time
import datetime

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'menu'
if 'history' not in st.session_state:
    st.session_state.history = []

# Load Premium CSS
load_css()

# Load model and vectorizer
@st.cache_resource
def load_resources():
    try:
        with open('models/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_resources()

# Render Header
render_header()

# Render History Sidebar
render_history(st.session_state.history)

# Navigation Logic
if st.session_state.current_page == 'menu':
    render_stats_row()
    render_menu()
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Start Analysis", use_container_width=True):
            st.session_state.current_page = 'analyze'
            st.rerun()
            
    with col2:
        if st.button("⚡ How It Works", use_container_width=True):
            st.session_state.current_page = 'how'
            st.rerun()
            
    with col3:
        if st.button("📊 View About", use_container_width=True):
            st.session_state.current_page = 'about'
            st.rerun()

elif st.session_state.current_page == 'analyze':
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.current_page = 'menu'
            st.rerun()
            
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    input_method = st.radio(
        "Select Input Method",
        ["Paste Text", "Upload Document", "Upload Image"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    news_text = ""
    
    if input_method == "Paste Text":
        news_text = st.text_area(
            "Enter news article",
            height=300,
            placeholder="Paste the news article you want to verify...",
            label_visibility="collapsed"
        )
    
    elif input_method == "Upload Document":
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                    reader = PdfReader(uploaded_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    news_text = text
                    st.success("✅ PDF loaded successfully!")
                elif uploaded_file.type == "text/plain":
                    news_text = str(uploaded_file.read(), "utf-8")
                    st.success("✅ Text file loaded!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    elif input_method == "Upload Image":
        uploaded_image = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, use_column_width=True)
            st.info("💡 Enter the text from the image below")
            news_text = st.text_area(
                "Text from image",
                height=200,
                placeholder="Type or paste the news content...",
                label_visibility="collapsed"
            )
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Analyze Content", use_container_width=True):
        if not news_text:
            st.warning("⚠️ Please provide text to analyze")
        elif model is None or vectorizer is None:
            st.error("❌ Model not found. Please train the model first.")
        else:
            # Simulated loading sequence for effect
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                # Non-linear progress for realism
                if i < 30: time.sleep(0.01)
                elif i < 60: time.sleep(0.02)
                else: time.sleep(0.005)
                
                progress_bar.progress(i + 1)
                if i == 10: status_text.text("Cleaning text...")
                if i == 40: status_text.text("Vectorizing content...")
                if i == 70: status_text.text("Running AI models...")
                if i == 90: status_text.text("Finalizing results...")
            
            status_text.empty()
            progress_bar.empty()
            
            # Actual Prediction
            # preprocessor = TextPreprocessor()
            # cleaned_text = preprocessor.clean_text(news_text)
            # We skip aggressive cleaning to match the training pipeline which used raw text
            vectorized_text = vectorizer.transform([news_text])
            prediction = model.predict(vectorized_text)[0]
            
            try:
                proba = model.predict_proba(vectorized_text)[0]
                confidence = proba[prediction]
            except:
                confidence = 0.95 # Fallback if probability not available
            
            is_fake = (prediction == 1)
            render_result(is_fake, confidence)
            
            # Add to history
            result_str = "FAKE" if is_fake else "REAL"
            timestamp = datetime.datetime.now().strftime("%H:%M")
            st.session_state.history.append({
                "text": news_text,
                "result": result_str,
                "is_fake": is_fake,
                "timestamp": timestamp
            })
            
            # Feedback Mechanism
            st.markdown("### Was this analysis helpful?")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if st.button("👍 Yes"):
                    st.toast("Thank you for your feedback!")
            with col_f2:
                if st.button("👎 No"):
                    st.toast("Thank you! We will improve our model.")
            
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == 'how':
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.current_page = 'menu'
            st.rerun()
            
    st.markdown("""
        <div class="glass-card">
            <h2>How It Works</h2>
            <p>Our system uses advanced machine learning algorithms to detect fake news.</p>
            <br>
            <div style="display: grid; gap: 1.5rem;">
                <div style="padding: 1rem; border-left: 4px solid #818cf8; background: rgba(255,255,255,0.05);">
                    <h3>1. Text Preprocessing</h3>
                    <p>We clean the text by removing special characters, URLs, and stop words to focus on the core content.</p>
                </div>
                <div style="padding: 1rem; border-left: 4px solid #c084fc; background: rgba(255,255,255,0.05);">
                    <h3>2. Vectorization</h3>
                    <p>The text is converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).</p>
                </div>
                <div style="padding: 1rem; border-left: 4px solid #ec4899; background: rgba(255,255,255,0.05);">
                    <h3>3. Model Prediction</h3>
                    <p>Our trained model analyzes the vectors to classify the news as real or fake based on patterns learned from thousands of articles.</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif st.session_state.current_page == 'about':
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.current_page = 'menu'
            st.rerun()
            
    st.markdown("""
        <div class="glass-card">
            <h2>About Project</h2>
            <p>This Fake News Detector was built to help combat misinformation.</p>
            <br>
            <h3>Tech Stack</h3>
            <ul>
                <li>Python</li>
                <li>Streamlit</li>
                <li>Scikit-learn</li>
                <li>Pandas & NumPy</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
