import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. DEPENDENCY FIX: CUSTOM MAPPING FUNCTION ---
# This dictionary and function MUST be present here so the saved pipeline
# can successfully load the FunctionTransformer step.
size_mapping = {
    '<10': 1, '10/49': 2, '50-99': 3, '100-500': 4, '500-999': 5,
    '1000-4999': 6, '5000-9999': 7, '10000+': 8, 'Unknown': 0
}
def map_company_size(arr):
    mapper = np.vectorize(lambda x: size_mapping.get(x, 0))
    # Reshaping is necessary as per the original pipeline
    return mapper(arr).reshape(-1, 1)


# --- 2. MODEL LOADING ---
try:
    # Load the best XGBoost pipeline saved from your training script
    model = joblib.load('job_change_predictor_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file not found. Ensure 'job_change_predictor_pipeline.pkl' is in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


st.set_page_config(page_title="Job Change Predictor", layout="wide")
st.title("Data Science Job Change Predictor") # Removed emoji
st.markdown(
    """
    **Model Performance:** Final Tuned XGBoost Accuracy on Test Set: **81.16%**
    ---
    """
)

# --- 3. UI INPUT FORM ---

# Helper function to get all 10 required features from the user
def get_user_inputs():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Applicant Details")
        
        # Numerical Features
        city_development_index = st.slider(
            "City Development Index (0.0 to 1.0)", 
            0.0, 1.0, 0.75, 0.01, help="Index of city development, standardized."
        )
        training_hours = st.number_input(
            "Total Training Hours Completed", 
            min_value=0, max_value=500, value=100
        )
        
        # Categorical Features
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        relevant_experience = st.selectbox(
            "Relevant Experience", 
            ['Has relevant experience', 'No relevant experience']
        )
        
    with col2:
        st.header("Education & Enrollment")

        experience = st.selectbox(
            "Years of Experience", 
            ['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10-20', '>20']
        )
        enrolled_university = st.selectbox(
            "Enrolled University Status", 
            ['no_enrollment', 'Full time course', 'Part time course']
        )
        education_level = st.selectbox(
            "Education Level", 
            ['Graduate', 'Masters', 'High School', 'Phd', 'Primary School']
        )
        
        major_discipline = st.selectbox(
            "Major Discipline", 
            ['STEM', 'Humanities', 'Other', 'Business Degree', 'Arts', 'No Major']
        )
        
    with col3:
        st.header("Company Details")
        
        company_type = st.selectbox(
            "Company Type", 
            ['Pvt Ltd', 'Funded Startup', 'Public Sector', 'Early Stage Startup', 'Other', 'NGO']
        )
        
        company_size = st.selectbox(
            "Company Size (Ordinal)", 
            ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+']
        )

    # Return inputs in the correct DataFrame column order (matching training data X)
    return {
        'city_development_index': city_development_index, 
        'experience': experience, 
        'training_hours': training_hours,
        'gender': gender, 
        'relevent_experience': relevant_experience,
        'enrolled_university': enrolled_university, 
        'education_level': education_level, 
        'major_discipline': major_discipline, 
        'company_type': company_type, 
        'company_size': company_size,
    }


# --- 4. PREDICTION LOGIC ---
user_data = get_user_inputs()

st.markdown("---")

if st.button("Predict Likelihood of Job Change", type="primary"):
    
    # Convert inputs to a DataFrame
    input_df = pd.DataFrame([user_data])
    
    with st.spinner('Calculating prediction...'):
        # Predict probability of the positive class (target=1)
        # The pipeline automatically handles all preprocessing (imputation, scaling, encoding, custom mapping)
        prediction_proba = model.predict_proba(input_df)[0][1]
        
    st.markdown("## Prediction Result")

    # Display results with visual cues
    if prediction_proba >= 0.5:
        st.error("HIGH LIKELIHOOD OF CANDIDATE SEEKING JOB CHANGE") # Removed emoji
        st.metric("Probability of Change (Target=1)", f"{prediction_proba * 100:.2f}%", delta_color="inverse")
    else:
        st.success("LOW LIKELIHOOD OF CANDIDATE SEEKING JOB CHANGE") # Removed emoji
        st.metric("Probability of Change (Target=1)", f"{prediction_proba * 100:.2f}%")
        
    st.caption(f"Confidence score for the current inputs: {abs(prediction_proba - 0.5) * 200:.2f}% (measured from the 50% threshold)")
