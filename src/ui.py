import streamlit as st
import base64
import os

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_css():
    """Injects the premium CSS styles."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
        
        :root {
            --primary: #F97316; /* Orange */
            --primary-dark: #EA580C;
            --secondary: #FB923C;
            --bg-dark: #0F172A;
            --card-bg: rgba(15, 23, 42, 0.8);
            --text-primary: #FFFFFF;
            --text-secondary: #94a3b8;
        }
        
        * {
            font-family: 'Outfit', sans-serif;
        }

        /* Modern Button Styles */
        div.stButton > button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            padding: 0.75rem 2rem !important;
            border-radius: 12px !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            letter-spacing: 0.5px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 6px rgba(249, 115, 22, 0.2) !important;
            width: 100% !important;
            text-transform: uppercase !important;
        }

        div.stButton > button:hover {
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 10px 20px rgba(249, 115, 22, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.2) !important;
            background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%) !important;
            border-color: rgba(255, 255, 255, 0.2) !important;
        }

        div.stButton > button:active {
            transform: translateY(0) scale(0.98) !important;
            box-shadow: 0 2px 4px rgba(249, 115, 22, 0.2) !important;
        }

        div.stButton > button:focus {
            outline: none !important;
            box-shadow: 0 0 0 3px rgba(249, 115, 22, 0.5) !important;
        }
        
        /* Neural Hex Grid Background */
        .stApp {
            background-color: #050b14;
            background-image: 
                radial-gradient(circle at 50% 50%, rgba(249, 115, 22, 0.08) 0%, transparent 50%),
                linear-gradient(0deg, transparent 24%, rgba(255, 255, 255, 0.03) 25%, rgba(255, 255, 255, 0.03) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, 0.03) 75%, rgba(255, 255, 255, 0.03) 76%, transparent 77%, transparent),
                linear-gradient(90deg, transparent 24%, rgba(255, 255, 255, 0.03) 25%, rgba(255, 255, 255, 0.03) 26%, transparent 27%, transparent 74%, rgba(255, 255, 255, 0.03) 75%, rgba(255, 255, 255, 0.03) 76%, transparent 77%, transparent);
            background-size: 100% 100%, 50px 50px, 50px 50px;
            animation: hexFlow 30s linear infinite;
        }
        
        /* Animated Neural Nodes */
        .stApp::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(2px 2px at 20px 30px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(2px 2px at 40px 70px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(2px 2px at 50px 160px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(2px 2px at 90px 40px, #ffffff, rgba(0,0,0,0)),
                radial-gradient(2px 2px at 130px 80px, #ffffff, rgba(0,0,0,0));
            background-repeat: repeat;
            background-size: 200px 200px;
            animation: twinkle 4s infinite linear;
            opacity: 0.3;
            pointer-events: none;
        }

        /* Moving Spotlight */
        .stApp::after {
            content: '';
            position: fixed;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle at center, rgba(59, 130, 246, 0.08) 0%, transparent 40%);
            animation: spotlightMove 15s infinite alternate ease-in-out;
            pointer-events: none;
            z-index: 0;
        }

        @keyframes hexFlow {
            0% { background-position: 0 0, 0 0, 0 0; }
            100% { background-position: 0 0, 0 50px, 50px 0; }
        }

        @keyframes twinkle {
            0% { transform: translateY(0); opacity: 0.3; }
            50% { opacity: 0.6; }
            100% { transform: translateY(-20px); opacity: 0.3; }
        }

        @keyframes spotlightMove {
            0% { transform: translate(0, 0); }
            100% { transform: translate(20px, 20px); }
        }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(15, 23, 42, 0.6);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            margin-bottom: 1.5rem;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            position: relative;
            overflow: hidden;
            z-index: 1;
        }
        
        .glass-card::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 24px;
            box-shadow: inset 0 0 20px rgba(249, 115, 22, 0.05);
            pointer-events: none;
        }
        
        .glass-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 50px -10px rgba(0, 0, 0, 0.5), 0 0 20px rgba(249, 115, 22, 0.2);
            border: 1px solid rgba(249, 115, 22, 0.4);
            background: rgba(15, 23, 42, 0.8);
        }

        /* Typography */
        h1, h2, h3 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }
        
        p, li {
            color: var(--text-secondary) !important;
            line-height: 1.6;
        }

        /* Custom Input Fields */
        .stTextArea textarea, .stTextInput input {
            background: rgba(10, 15, 30, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            color: white !important;
            padding: 1rem !important;
            transition: all 0.3s ease;
        }
        
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.2), 0 0 15px rgba(249, 115, 22, 0.1) !important;
        }

        /* Results */
        .result-card {
            padding: 3rem;
            border-radius: 16px;
            text-align: center;
            animation: fadeIn 0.6s ease-out;
            margin-top: 2rem;
            background: rgba(0,0,0,0.8);
            border: 1px solid rgba(255,255,255,0.1);
            position: relative;
            z-index: 1;
        }
        
        .result-fake {
            border-color: #ef4444;
            box-shadow: 0 0 50px rgba(239, 68, 68, 0.15);
        }
        
        .result-real {
            border-color: #22c55e;
            box-shadow: 0 0 50px rgba(34, 197, 94, 0.15);
        }
        
        /* Gauge Animation */
        .gauge-container {
            position: relative;
            width: 200px;
            height: 100px;
            margin: 0 auto 2rem auto;
            overflow: hidden;
        }
        
        .gauge-body {
            position: absolute;
            top: 0;
            left: 0;
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.05);
            box-sizing: border-box;
            border: 20px solid rgba(255, 255, 255, 0.05);
            border-bottom-color: transparent;
            border-left-color: transparent;
            transform: rotate(45deg);
        }
        
        .gauge-fill {
            position: absolute;
            top: 0;
            left: 0;
            width: 200px;
            height: 200px;
            border-radius: 50%;
            box-sizing: border-box;
            border: 20px solid transparent;
            border-top-color: var(--gauge-color);
            border-right-color: var(--gauge-color);
            transform: rotate(45deg);
            transition: transform 1.5s cubic-bezier(0.4, 0, 0.2, 1);
            z-index: 1;
            filter: drop-shadow(0 0 10px var(--gauge-color));
        }
        
        .gauge-cover {
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 160px;
            height: 80px;
            background: #0F172A; /* Match background fallback */
            z-index: 2;
            /* Hack to make cover transparent to gradient but opaque to gauge */
            background: radial-gradient(circle at 50% 0, transparent 0%, #0F172A 100%); 
            /* Actually just matching the dark bg is safer */
            background: #050b14; 
        }
        
        .gauge-text {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            font-size: 2.5rem;
            font-weight: 800;
            color: white;
            z-index: 3;
            line-height: 1;
            text-shadow: 0 0 20px rgba(255,255,255,0.5);
        }

        /* Icon Animations */
        .menu-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            display: inline-block;
            text-shadow: 0 0 20px rgba(249, 115, 22, 0.3);
        }

        .glass-card:hover .menu-icon {
            transform: scale(1.2) translateY(-5px);
            text-shadow: 0 0 30px rgba(249, 115, 22, 0.8);
        }

    </style>
    """, unsafe_allow_html=True)

def render_header():
    # Try to load the local logo, fallback to placeholder if missing
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        img_b64 = get_base64_of_bin_file(logo_path)
        img_src = f"data:image/png;base64,{img_b64}"
    else:
        # Fallback placeholder
        img_src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    st.markdown(f"""
        <style>
            @keyframes logoFloat {{
                0% {{ transform: translateY(0); filter: drop-shadow(0 0 20px rgba(249, 115, 22, 0.3)); }}
                50% {{ transform: translateY(-10px); filter: drop-shadow(0 0 40px rgba(249, 115, 22, 0.6)); }}
                100% {{ transform: translateY(0); filter: drop-shadow(0 0 20px rgba(249, 115, 22, 0.3)); }}
            }}
        </style>
        <div class="main-header" style="text-align: center; padding: 2rem 0;">
            <img src="{img_src}" 
                 style="width: 150px; height: 150px; object-fit: contain; animation: logoFloat 4s ease-in-out infinite; margin-bottom: 1rem; border-radius: 50%;">
            <h1 style="font-size: 4rem; background: linear-gradient(to right, #F97316, #FB923C, #fff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; font-weight: 800;">
                TruthLens AI
            </h1>
            <p style="font-size: 1.4rem; color: #94a3b8; max-width: 600px; margin: 0 auto; font-weight: 300;">
                <span style="color: #F97316;">●</span> Fake News Detector
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_stats_row():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 1.5rem;">
                <h3 style="color: #F97316 !important; font-size: 2.5rem; margin: 0;">98.2%</h3>
                <p style="margin: 0; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem;">Accuracy Rate</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 1.5rem;">
                <h3 style="color: #3B82F6 !important; font-size: 2.5rem; margin: 0;">50k+</h3>
                <p style="margin: 0; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem;">Articles Analyzed</p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 1.5rem;">
                <h3 style="color: #22C55E !important; font-size: 2.5rem; margin: 0;">&lt;1s</h3>
                <p style="margin: 0; text-transform: uppercase; letter-spacing: 1px; font-size: 0.8rem;">Processing Time</p>
            </div>
        """, unsafe_allow_html=True)

def render_menu():
    st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
            <div class="glass-card" style="cursor: pointer; text-align: center;" onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'analyze'}, '*')">
                <div class="menu-icon">🔍</div>
                <h3>Analyze News</h3>
                <p>Detect fake news in real-time</p>
            </div>
            <div class="glass-card" style="cursor: pointer; text-align: center;" onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'how'}, '*')">
                <div class="menu-icon">⚡</div>
                <h3>How It Works</h3>
                <p>Learn about our AI technology</p>
            </div>
            <div class="glass-card" style="cursor: pointer; text-align: center;" onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'about'}, '*')">
                <div class="menu-icon">📊</div>
                <h3>About Project</h3>
                <p>Meet the team & mission</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_result(is_fake, confidence):
    color = "#ef4444" if is_fake else "#22c55e"
    result_text = "FAKE NEWS DETECTED" if is_fake else "REAL NEWS VERIFIED"
    icon = "⚠️" if is_fake else "✅"
    
    gauge_rotation = int((confidence * 180) - 180 + 45) 
    if is_fake:
        gauge_color = "#ef4444"
    else:
        gauge_color = "#22c55e"

    st.markdown(f"""
        <style>
            :root {{
                --gauge-color: {gauge_color};
            }}
            .gauge-fill {{
                transform: rotate({gauge_rotation}deg) !important;
            }}
        </style>
        <div class="result-card {'result-fake' if is_fake else 'result-real'}">
            <div class="gauge-container">
                <div class="gauge-body"></div>
                <div class="gauge-fill"></div>
                <div class="gauge-cover"></div>
                <div class="gauge-text">{confidence:.1%}</div>
            </div>
            <h2 style="color: {color} !important; font-size: 2.5rem; margin-bottom: 1rem;">
                {icon} {result_text}
            </h2>
            <p style="font-size: 1.2rem; opacity: 0.8;">
                Our AI is <strong>{confidence:.1%}</strong> confident in this assessment.
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_history(history):
    st.sidebar.markdown("### 🕒 Recent Analysis")
    if not history:
        st.sidebar.info("No analysis yet.")
        return

    for item in reversed(history[-5:]): # Show last 5
        color = "#ef4444" if item['is_fake'] else "#22c55e"
        icon = "⚠️" if item['is_fake'] else "✅"
        st.sidebar.markdown(f"""
            <div style="padding: 0.8rem; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {color};">
                <div style="font-weight: 600; font-size: 0.9rem; color: white;">{icon} {item['result']}</div>
                <div style="font-size: 0.8rem; color: #94a3b8;">{item['timestamp']}</div>
                <div style="font-size: 0.75rem; color: #64748b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{item['text'][:30]}...</div>
            </div>
        """, unsafe_allow_html=True)
