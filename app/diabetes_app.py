import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd

# Load model and scaler
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetes_xgb_model.pkl')
model = joblib.load(model_path)
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'diabetes_scaler.pkl')
scaler = joblib.load(scaler_path)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
        font-size: 24px;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
        font-size: 24px;
    }
    .risk-low {
        color: #2ecc71;
        font-weight: bold;
        font-size: 24px;
    }
    .feature-importance {
        margin-top: 20px;
        padding: 15px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("Diabetes Risk Assessment Tool")
st.markdown("""
This tool assesses your risk of having diabetes based on health metrics. 
Please provide the following information for an accurate assessment.
""")

# Input form
with st.form("diabetes_form"):
    st.header("Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=100)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    with col2:
        insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
    
    submitted = st.form_submit_button("Assess Diabetes Risk")

if submitted:
    # Create feature array
    input_data = np.array([
        pregnancies, glucose, blood_pressure, skin_thickness, 
        insulin, bmi, diabetes_pedigree, age
    ]).reshape(1, -1)
    
    # Add engineered features
    glucose_bmi_ratio = glucose / bmi if bmi != 0 else 0
    age_glucose = age * glucose
    bp_glucose = blood_pressure * glucose
    input_data = np.append(input_data, [[glucose_bmi_ratio, age_glucose, bp_glucose]], axis=1)
    
    # Scale features
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1]
    
    # Display results
    st.header("Assessment Results")
    
    if probability >= 0.7:
        st.markdown(f'<p class="risk-high">High Risk ({probability*100:.1f}% probability)</p>', 
                   unsafe_allow_html=True)
        st.warning("This suggests a high likelihood of diabetes. Please consult with a healthcare professional for further evaluation and possible testing.")
    elif probability >= 0.4:
        st.markdown(f'<p class="risk-medium">Moderate Risk ({probability*100:.1f}% probability)</p>', 
                   unsafe_allow_html=True)
        st.info("This suggests some risk factors for diabetes. Consider lifestyle changes and monitoring your health indicators.")
    else:
        st.markdown(f'<p class="risk-low">Low Risk ({probability*100:.1f}% probability)</p>', 
                   unsafe_allow_html=True)
        st.success("Your current indicators suggest low risk for diabetes. Maintain healthy habits for prevention.")
    
    # Detailed breakdown
    st.subheader("Risk Factors Analysis")
    
    # Create a DataFrame for visualization
    factors = {
        'Factor': ['Glucose Level', 'BMI', 'Age', 'Blood Pressure', 
                  'Insulin Level', 'Pregnancies', 'Family History'],
        'Value': [glucose, bmi, age, blood_pressure, insulin, pregnancies, diabetes_pedigree],
        'Normal Range': ['70-99 mg/dL', '18.5-24.9', '-', '<120/<80', '2.6-24.9 μU/mL', '-', '-']
    }
    factors_df = pd.DataFrame(factors)
    
    # Highlight concerning values
    def highlight_risk(row):
        if row['Factor'] == 'Glucose Level' and row['Value'] > 126:
            return ['background-color: #ffcccc'] * len(row)
        elif row['Factor'] == 'BMI' and row['Value'] >= 30:
            return ['background-color: #ffcccc'] * len(row)
        elif row['Factor'] == 'Blood Pressure' and row['Value'] >= 130:
            return ['background-color: #ffcccc'] * len(row)
        elif row['Factor'] == 'Insulin Level' and row['Value'] > 25:
            return ['background-color: #ffcccc'] * len(row)
        else:
            return [''] * len(row)
    
    st.dataframe(factors_df.style.apply(highlight_risk, axis=1))
    
    # Recommendations
    st.subheader("Personalized Recommendations")
    if glucose > 126:
        st.write("🔴 Your glucose level is above normal (fasting glucose >126 mg/dL suggests diabetes)")
    if bmi >= 30:
        st.write("🔴 Your BMI indicates obesity, a significant diabetes risk factor)")
    if blood_pressure >= 130:
        st.write("🔴 Elevated blood pressure is associated with higher diabetes risk)")
    
    general_advice = [
        "✅ Maintain a balanced diet rich in whole grains, fruits, and vegetables",
        "✅ Engage in regular physical activity (150+ minutes per week)",
        "✅ Monitor your blood sugar levels if you have risk factors",
        "✅ Maintain a healthy weight",
        "✅ Get regular health check-ups"
    ]
    
    for advice in general_advice:
        st.write(advice)
    
    # Disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer:** This tool provides a risk assessment based on statistical patterns and should not replace professional medical advice. 
    Always consult with a healthcare provider for diagnosis and treatment.
    """)