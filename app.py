# ========================================
# BREAST CANCER PREDICTION WEB APP
# ========================================
# Run with: streamlit run app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .benign {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .malignant {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model components
@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('models/breast_cancer_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please train the model first.")
        return None, None

# Feature names (in correct order)
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Feature ranges (typical values for input validation)
FEATURE_RANGES = {
    'radius_mean': (6.0, 30.0, 14.0),
    'texture_mean': (9.0, 40.0, 19.0),
    'perimeter_mean': (43.0, 190.0, 92.0),
    'area_mean': (143.0, 2500.0, 655.0),
    'smoothness_mean': (0.05, 0.17, 0.10),
    'compactness_mean': (0.02, 0.35, 0.10),
    'concavity_mean': (0.0, 0.43, 0.09),
    'concave points_mean': (0.0, 0.20, 0.05),
    'symmetry_mean': (0.10, 0.30, 0.18),
    'fractal_dimension_mean': (0.05, 0.10, 0.06),
    'radius_se': (0.1, 3.0, 0.4),
    'texture_se': (0.3, 5.0, 1.2),
    'perimeter_se': (0.7, 22.0, 2.9),
    'area_se': (6.0, 542.0, 40.0),
    'smoothness_se': (0.001, 0.03, 0.007),
    'compactness_se': (0.002, 0.14, 0.025),
    'concavity_se': (0.0, 0.40, 0.03),
    'concave points_se': (0.0, 0.05, 0.01),
    'symmetry_se': (0.007, 0.08, 0.02),
    'fractal_dimension_se': (0.0008, 0.03, 0.004),
    'radius_worst': (7.0, 37.0, 16.0),
    'texture_worst': (12.0, 50.0, 25.0),
    'perimeter_worst': (50.0, 252.0, 107.0),
    'area_worst': (185.0, 4254.0, 881.0),
    'smoothness_worst': (0.07, 0.23, 0.13),
    'compactness_worst': (0.02, 1.06, 0.25),
    'concavity_worst': (0.0, 1.25, 0.27),
    'concave points_worst': (0.0, 0.29, 0.11),
    'symmetry_worst': (0.15, 0.66, 0.29),
    'fractal_dimension_worst': (0.055, 0.21, 0.08)
}

def predict_cancer(features, model, scaler):
    """Make prediction"""
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    return prediction, probabilities

def create_gauge_chart(probability):
    """Create a gauge chart for probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Malignancy Risk", 'font': {'size': 24}},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# Main app
def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>Breast Cancer Prediction System</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-powered diagnostic assistance using machine learning</p>", 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, scaler = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            """
            This application uses a Logistic Regression model trained on the 
            Wisconsin Breast Cancer Dataset to predict whether a tumor is 
            benign or malignant based on cell nucleus characteristics.
            
            **Model Performance:**
            - Accuracy: 97.37%
            - Precision: 97.62%
            - Recall: 95.35%
            """
        )
        
        st.header("Input Method")
        input_method = st.radio(
            "Choose input method:",
            ["Manual Input", "Upload CSV", "Use Sample Data"]
        )
    
    # Main content
    if input_method == "Manual Input":
        st.header("Enter Cell Nucleus Measurements")
        
        # Create tabs for different feature groups
        tab1, tab2, tab3 = st.tabs(["Mean Values", "Standard Error", "Worst Values"])
        
        features = []
        
        with tab1:
            st.subheader("Mean Measurements")
            cols = st.columns(2)
            for i, feature in enumerate(FEATURE_NAMES[:10]):
                min_val, max_val, default = FEATURE_RANGES[feature]
                with cols[i % 2]:
                    value = st.number_input(
                        feature.replace('_', ' ').title(),
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default),
                        key=f"mean_{i}"
                    )
                    features.append(value)
        
        with tab2:
            st.subheader("Standard Error Measurements")
            cols = st.columns(2)
            for i, feature in enumerate(FEATURE_NAMES[10:20]):
                min_val, max_val, default = FEATURE_RANGES[feature]
                with cols[i % 2]:
                    value = st.number_input(
                        feature.replace('_', ' ').title(),
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default),
                        key=f"se_{i}"
                    )
                    features.append(value)
        
        with tab3:
            st.subheader("Worst Value Measurements")
            cols = st.columns(2)
            for i, feature in enumerate(FEATURE_NAMES[20:30]):
                min_val, max_val, default = FEATURE_RANGES[feature]
                with cols[i % 2]:
                    value = st.number_input(
                        feature.replace('_', ' ').title(),
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default),
                        key=f"worst_{i}"
                    )
                    features.append(value)
        
        # Predict button
        if st.button("Predict", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                prediction, probabilities = predict_cancer(features, model, scaler)
                
                st.markdown("---")
                st.header("🎯 Prediction Results")
                
                # Results in columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if prediction == 0:
                        st.markdown(
                            """<div class='result-box benign'>
                            <h2>✅ BENIGN</h2>
                            <p>The tumor is predicted to be benign (non-cancerous)</p>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """<div class='result-box malignant'>
                            <h2>⚠️ MALIGNANT</h2>
                            <p>The tumor is predicted to be malignant (cancerous)</p>
                            </div>""",
                            unsafe_allow_html=True
                        )
                    
                    st.metric("Confidence", f"{max(probabilities) * 100:.2f}%")
                    
                    # Probability breakdown
                    st.subheader("Probability Breakdown")
                    prob_df = pd.DataFrame({
                        'Diagnosis': ['Benign', 'Malignant'],
                        'Probability': [probabilities[0] * 100, probabilities[1] * 100]
                    })
                    fig = px.bar(prob_df, x='Diagnosis', y='Probability', 
                                color='Diagnosis',
                                color_discrete_map={'Benign': '#28a745', 'Malignant': '#dc3545'})
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Gauge chart
                    st.plotly_chart(create_gauge_chart(probabilities[1]), 
                                  use_container_width=True)
                    
                    # Risk level
                    malignant_prob = probabilities[1]
                    if malignant_prob < 0.3:
                        risk = "🟢 Low Risk"
                        risk_color = "green"
                    elif malignant_prob < 0.7:
                        risk = "🟡 Medium Risk"
                        risk_color = "orange"
                    else:
                        risk = "🔴 High Risk"
                        risk_color = "red"
                    
                    st.markdown(f"### Risk Level: <span style='color:{risk_color}'>{risk}</span>", 
                              unsafe_allow_html=True)
                
                # Disclaimer
                st.warning(
                    """
                    ⚕️ **Medical Disclaimer:** This prediction is for educational purposes only. 
                    It should NOT be used as a substitute for professional medical diagnosis. 
                    Always consult with qualified healthcare professionals for medical decisions.
                    """
                )
    
    elif input_method == "Upload CSV":
        st.header("📤 Upload CSV File")
        st.info("Upload a CSV file with 30 features in the correct order")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("Predict All"):
                predictions = []
                for _, row in df.iterrows():
                    features = row.values[:30]  # First 30 columns
                    pred, probs = predict_cancer(features, model, scaler)
                    predictions.append({
                        'Prediction': 'Benign' if pred == 0 else 'Malignant',
                        'Benign_Probability': f"{probs[0]*100:.2f}%",
                        'Malignant_Probability': f"{probs[1]*100:.2f}%"
                    })
                
                results_df = pd.DataFrame(predictions)
                st.write("Predictions:")
                st.dataframe(results_df)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    key='download-csv'
                )
    
    else:  # Use Sample Data
        st.header("📋 Sample Data Prediction")
        
        # Sample data (malignant case)
        sample_malignant = [
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
            0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
            0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
            0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]
        
        # Sample data (benign case)
        sample_benign = [
            13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781,
            0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146,
            0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2,
            0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
        ]
        
        sample_choice = st.selectbox(
            "Choose a sample:",
            ["Malignant Sample", "Benign Sample"]
        )
        
        sample_data = sample_malignant if sample_choice == "Malignant Sample" else sample_benign
        
        # Display sample data
        sample_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Value': sample_data
        })
        st.dataframe(sample_df, height=400)
        
        if st.button("🔍 Predict Sample", type="primary"):
            prediction, probabilities = predict_cancer(sample_data, model, scaler)
            
            col1, col2 = st.columns(2)
            with col1:
                result_text = "BENIGN ✅" if prediction == 0 else "MALIGNANT ⚠️"
                st.markdown(f"### Prediction: {result_text}")
                st.metric("Confidence", f"{max(probabilities) * 100:.2f}%")
            
            with col2:
                st.plotly_chart(create_gauge_chart(probabilities[1]), 
                              use_container_width=True)

if __name__ == "__main__":
    main()