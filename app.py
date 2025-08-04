
import streamlit as st
import numpy as np
import joblib
import pandas as pd
import json
import random

# Load model and scaler
try:
    model = joblib.load("best_fire_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("🔥 Model or scaler file not found! Ensure 'best_fire_detection_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Set page configuration
st.set_page_config(page_title="Fire Type Quest", layout="wide", initial_sidebar_state="expanded")

# Custom CSS with fire-themed Tailwind styling
st.markdown("""
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(to bottom, #ff6b6b, #ffe066);
        }
        .stButton>button {
            background: linear-gradient(to right, #ff4b2b, #ff416c);
            color: white;
            border-radius: 0.75rem;
            padding: 0.75rem 1.5rem;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(255, 75, 43, 0.5);
        }
        .stNumberInput input, .stSlider input {
            border-radius: 0.5rem;
            border: 2px solid #f97316;
            padding: 0.5rem;
            background-color: #fff7ed;
        }
        .stSelectbox select {
            border-radius: 0.5rem;
            border: 2px solid #f97316;
            padding: 0.5rem;
            background-color: #fff7ed;
        }
        .prediction-box {
            background: linear-gradient(to right, #fefcbf, #fed7aa);
            border: 3px solid #f97316;
            border-radius: 1rem;
            padding: 1rem;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        .sidebar .sidebar-content {
            background-color: #fff7ed;
            border-right: 2px solid #f97316;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Fun facts and quotes
fun_facts = [
    "🔥 Did you know? Vegetation fires can release as much CO2 as a small country in a single season!",
    "🌊 Offshore fires are rare but can occur on oil platforms, releasing intense heat!",
    "🏭 Static land sources include industrial fires, often detectable by high FRP values!"
]
quotes = [
    "Fight fire with knowledge! – Fire Science Proverb",
    "Every flame tells a story. – Nature Enthusiast",
    "Protect our planet, one prediction at a time! – Eco Warrior"
]

# Sidebar with engaging content
with st.sidebar:
    st.markdown('<h1 class="text-2xl font-bold text-orange-600">🔥 Fire Type Quest</h1>', unsafe_allow_html=True)
    st.markdown("Embark on a mission to classify fires using MODIS data! 🚀")
    st.markdown("### 📋 Input Guide")
    st.markdown("""
    - **Brightness**: 300–450 K (hotter than a summer day! ☀️)
    - **Brightness T31**: 290–400 K (cooler side of fire 🌡️)
    - **FRP**: 0–100 MW (fire's power level 💥)
    - **Scan/Track**: 1–2 (pixel size, like zooming in 🔍)
    - **Confidence**: Low, Nominal, High (how sure are you? 🤔)
    """)
    if st.session_state.history:
        st.markdown("### 🕰️ Quest History")
        for idx, entry in enumerate(st.session_state.history):
            st.markdown(f"**Quest {idx+1}:** {entry['result']} (Confidence: {entry['confidence']:.2%})")

# Main app layout
st.markdown('<h1 class="text-4xl font-extrabold text-center mb-6 text-red-600">🔥 Fire Type Quest 🔥</h1>', unsafe_allow_html=True)
st.markdown('<p class="text-center text-orange-800 mb-4">Join the adventure! Enter MODIS readings to uncover the fire type!</p>', unsafe_allow_html=True)

# Input form with sliders
with st.form("input_form"):
    st.markdown('<h2 class="text-2xl font-semibold text-orange-700 mb-4">🛠️ Input Your Data</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        brightness = st.slider("🔥 Brightness (K)", min_value=0.0, max_value=1000.0, value=300.0, step=0.1, help="Brightness temperature in Kelvin")
        bright_t31 = st.slider("🌡️ Brightness T31 (K)", min_value=0.0, max_value=1000.0, value=290.0, step=0.1, help="Brightness temperature at T31 channel")
        frp = st.slider("💥 FRP (MW)", min_value=0.0, max_value=1000.0, value=15.0, step=0.1, help="Fire radiative power in megawatts")
    
    with col2:
        scan = st.slider("🔍 Scan", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Scan pixel size")
        track = st.slider("📏 Track", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Track pixel size")
        confidence = st.selectbox("🤔 Confidence Level", ["low", "nominal", "high"], help="Confidence level of detection")

    col_submit, col_reset = st.columns(2)
    with col_submit:
        submitted = st.form_submit_button("🔥 Predict Fire Type")
    with col_reset:
        reset = st.form_submit_button("🗑️ Reset Inputs")

# Reset inputs
if reset:
    st.session_state.brightness = 300.0
    st.session_state.bright_t31 = 290.0
    st.session_state.frp = 15.0
    st.session_state.scan = 1.0
    st.session_state.track = 1.0
    st.session_state.confidence = "low"
    st.experimental_rerun()

# Live input preview
st.markdown("### 📊 Live Input Preview")
input_data = pd.DataFrame({
    "Feature": ["Brightness", "Brightness T31", "FRP", "Scan", "Track", "Confidence"],
    "Value": [brightness, bright_t31, frp, scan, track, confidence]
})
st.table(input_data)

# Process prediction
if submitted:
    # Map confidence to numeric
    confidence_map = {"low": 0, "nominal": 1, "high": 2}
    confidence_val = confidence_map[confidence]

    # Input validation
    warnings = []
    if brightness < 300 or brightness > 450:
        warnings.append("⚠️ Brightness outside 300–450 K!")
    if bright_t31 < 290 or bright_t31 > 400:
        warnings.append("⚠️ Brightness T31 outside 290–400 K!")
    if frp < 0 or frp > 100:
        warnings.append("⚠️ FRP outside 0–100 MW!")
    if warnings:
        st.warning(" ".join(warnings))

    # Combine and scale input
    input_array = np.array([[brightness, bright_t31, frp, scan, track, confidence_val]])
    try:
        scaled_input = scaler.transform(input_array)
    except Exception as e:
        st.error(f"🚨 Error scaling input data: {str(e)}")
        st.stop()

    # Predict with debugging
    try:
        prediction = model.predict(scaled_input)[0]
        st.write(f"Debug: Raw Prediction = {prediction}")  # Debug output
        try:
            prediction_proba = model.predict_proba(scaled_input)[0]
            st.write(f"Debug: Probabilities = {prediction_proba}")  # Debug probabilities
            max_proba = max(prediction_proba)
        except AttributeError:
            st.write("Debug: Model does not support predict_proba")
            max_proba = None

        fire_types = {
            0: "Vegetation Fire 🌿🔥",
            2: "Static Land Source 🏭🔥",
            3: "Offshore Fire 🌊🔥"
        }
        result = fire_types.get(prediction, "Unknown Fire Type ❓")

        # Display prediction with animation
        st.markdown(f'<div class="prediction-box"><p class="text-2xl font-bold text-red-600">Predicted Fire Type: {result}</p></div>', unsafe_allow_html=True)
        if max_proba:
            st.markdown(f'<p class="text-orange-800">Confidence: {max_proba:.2%}</p>', unsafe_allow_html=True)
            st.progress(max_proba)

        # Save to history
        st.session_state.history.append({"result": result, "confidence": max_proba if max_proba else 1.0})
        if len(st.session_state.history) > 10:
            st.session_state.history.pop(0)

        # Fun fact and quote
        st.markdown("### 🎉 Fun Fact")
        st.markdown(f'<p class="text-orange-700">{random.choice(fun_facts)}</p>', unsafe_allow_html=True)
        st.markdown("### 💡 Inspiration")
        st.markdown(f'<p class="text-orange-700 italic">{random.choice(quotes)}</p>', unsafe_allow_html=True)

        # Radar chart for input visualization with fallback
        st.markdown("### 📈 Input Visualization (Radar Chart)")
        chart_data = {
            "labels": ["Brightness", "Brightness T31", "FRP", "Scan", "Track"],
            "values": [min(brightness / 1000, 1), min(bright_t31 / 1000, 1), min(frp / 1000, 1), min(scan / 10, 1), min(track / 10, 1)]
        }
        try:
            st.markdown('<canvas id="radarChart" class="mt-4"></canvas>', unsafe_allow_html=True)
            st.markdown(f"""
                <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
                <script>
                    const ctx = document.getElementById('radarChart').getContext('2d');
                    new Chart(ctx, {{
                        type: 'radar',
                        data: {{
                            labels: {json.dumps(chart_data["labels"])},
                            datasets: [{{
                                label: 'Input Values',
                                data: {json.dumps(chart_data["values"])},
                                backgroundColor: 'rgba(255, 107, 107, 0.2)',
                                borderColor: '#ff6b6b',
                                borderWidth: 2,
                                pointBackgroundColor: '#ff416c'
                            }}]
                        }},
                        options: {{
                            scales: {{
                                r: {{
                                    beginAtZero: true,
                                    max: 1,
                                    ticks: {{ stepSize: 0.2 }}
                                }}
                            }},
                            plugins: {{
                                legend: {{ display: false }}
                            }}
                        }}
                    }});
                </script>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"🚨 Chart failed to load: {str(e)}. Showing data instead.")
            st.write("Chart Data:", chart_data)

    except Exception as e:
        st.error(f"🚨 Error during prediction: {str(e)}")

# Footer
st.markdown('<hr class="my-6 border-orange-300">', unsafe_allow_html=True)
st.markdown('<p class="text-center text-orange-600">Built with Streamlit & Tailwind CSS | © 2025 Fire Type Quest 🔥</p>', unsafe_allow_html=True)
