import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page Configuration
st.set_page_config(
    page_title="Decision Tree Classifier",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling and color combinations
st.markdown("""
    <style>
    /* Main theme background and colors */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    
    /* Header Container styling */
    .header-container {
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        color: #10B981;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #9CA3AF;
        font-size: 1rem;
    }
    
    /* Input Box styling */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background-color: #1F2937 !important;
        border-color: #374151 !important;
        color: #F3F4F6 !important;
        border-radius: 8px !important;
    }

    /* Custom Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: #FFFFFF;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4);
        transform: translateY(-2px);
    }

    /* Custom Result Card */
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    .result-yes {
        background-color: rgba(16, 185, 129, 0.15);
        border: 2px solid #10B981;
        color: #34D399;
    }
    .result-no {
        background-color: rgba(239, 68, 68, 0.15);
        border: 2px solid #EF4444;
        color: #F87171;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained pickle model
@st.cache_resource
def load_model():
    with open("DecisionTree.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model 'DecisionTree.pkl': {e}")
    st.stop()

# Header Section
st.markdown("""
    <div class="header-container">
        <div class="header-title">Predictive Intelligence Model</div>
        <div class="header-subtitle">Provide demographic and financial input to generate accurate predictions</div>
    </div>
""", unsafe_allow_html=True)

# Main Form Container
with st.form("prediction_form"):
    st.subheader("📋 Input Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        region = st.selectbox("Region", options=["North", "South", "East", "West", "Urban", "Suburban", "Rural"])
        
    with col2:
        occupation = st.selectbox("Occupation", options=["Salaried", "Self-Employed", "Business", "Student", "Other"])
        income = st.number_input("Income", min_value=0, max_value=10000000, value=50000, step=1000)

    submit_button = st.form_submit_button(label="⚡ Predict Outcome")

# Prediction handling logic
if submit_button:
    # Construct input dataframe with exact features extracted from the pkl
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Region": region,
        "Occupation": occupation,
        "Income": income
    }])

    try:
        # Generate prediction
        prediction = model.predict(input_df)[0]
        
        # Display Result
        st.markdown("---")
        if str(prediction).lower() == "yes":
            st.markdown(
                f'<div class="result-card result-yes">🎉 Prediction: <strong>YES</strong></div>', 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-card result-no">⚠️ Prediction: <strong>NO</strong></div>', 
                unsafe_allow_html=True
            )
            
    except Exception as err:
        st.error(f"Error during prediction: {err}")
        st.info("Note: If your raw model expects numeric target-encoded inputs instead of strings, ensure categorical fields match the exact format used during training.")
