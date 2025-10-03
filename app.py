import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. DEPENDENCY FIX: CUSTOM MAPPING FUNCTIONS (UPDATED) ---
# Dictionaries and functions must be present here so the saved pipeline can load.
size_mapping = {
    '<10': 1, '10/49': 2, '50-99': 3, '100-500': 4, '500-999': 5,
    '1000-4999': 6, '5000-9999': 7, '10000+': 8, 'Unknown': 0
}
# NEW MAPPING: Converts Experience strings to numerical values the imputer expects
experience_mapping = {
    '<1': 0.5, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, 
    '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, 
    '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, 
    '10-20': 15, '>20': 25, '20.0': 20 
}
# We use the raw experience strings here as input:
experience_options = ['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10-20', '>20']


def map_company_size(arr):
    mapper = np.vectorize(lambda x: size_mapping.get(x, 0))
    return mapper(arr).reshape(-1, 1)


# --- 2. MODEL LOADING ---
try:
    model = joblib.load('job_change_predictor_pipeline.pkl')
except FileNotFoundError:
    st.error("Model file not found. Ensure 'job_change_predictor_pipeline.pkl' is in the same folder.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


st.set_page_config(page_title="Job Change Predictor", layout="wide")
st.title("Data Science Job Predictor")
st.markdown(
    """
    **Model Performance:** Final Tuned XGBoost Accuracy on Test Set: **81.16%**
    ---
    """
)

# --- 3. UI INPUT FORM ---

def get_user_inputs():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Applicant Details")
        
        # Numerical Features (City Index and Training Hours)
        city_development_index = st.slider(
            "City Development Index (0.0 to 1.0)", 
            0.0, 1.0, 0.75, 0.01
        )
        training_hours = st.number_input(
            "Total Training Hours Completed", 
            min_value=0, max_value=500, value=100
        )
        
        # Categorical Features
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        relevant_experience = st.selectbox(
            "Relevant Experience", 
            ['Has relevent experience', 'No relevent experience']
        )
        
    with col2:
        st.header("Education & Enrollment")

        # ORNDINAL INPUT - RAW STRING
        experience = st.selectbox(
            "Years of Experience", 
            experience_options # Use the defined options
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

    # Returns the dictionary with raw data
    return {
        'city_development_index': city_development_index, 
        'experience_raw': experience, # Store raw string temporarily
        'training_hours': training_hours,
        'gender': gender, 
        'relevent_experience': relevant_experience,
        'enrolled_university': enrolled_university, 
        'education_level': education_level, 
        'major_discipline': major_discipline, 
        'company_type': company_type, 
        'company_size': company_size,
    }


# --- 4. PREDICTION LOGIC (FINALIZED FIX) ---
user_data_raw = get_user_inputs()

st.markdown("---")

if st.button("Predict Likelihood of Job Change", type="primary"):
    
    # 1. FIX THE DATA TYPE: Map the experience string to a number (as expected by the pipeline)
    exp_str = user_data_raw.pop('experience_raw') 
    user_data_raw['experience'] = experience_mapping.get(exp_str, 0.0) 

    # 2. Convert inputs to a DataFrame
    # Note: We must ensure the column order matches the training data exactly.
    input_df = pd.DataFrame([user_data_raw])
    
    # 3. CRITICAL: Reorder columns to ensure they match X_train order
    # (Though not always necessary, this eliminates the possibility of column mismatch errors)
    # Based on your notebook: city_development_index, gender, relevent_experience, enrolled_university, education_level, major_discipline, experience, company_size, company_type, training_hours
    
    # Create the correct column list based on your notebook's X_train definition (excluding the dropped 'city' and 'enrollee_id')
    # The order of features in the transformer is: city_development_index, experience, training_hours (numeric)
    # followed by nominal and ordinal features. We use the original training column order for the DataFrame.
    
    # This is the order of features present in your original DataFrame X:
    expected_columns = [
        'city_development_index', 'gender', 'relevent_experience', 
        'enrolled_university', 'education_level', 'major_discipline', 
        'experience', 'company_size', 'company_type', 'training_hours'
    ]
    
    # Reorder the input DataFrame columns to match the expected training order
    # We must also ensure ALL columns are present, even if some were not set by the user.
    # The dictionary creation automatically ensures columns are present, but we re-order here.
    try:
        input_df = input_df[expected_columns]
    except KeyError as e:
        st.error(f"Input Data Error: Missing expected column in input data frame. {e}")
        st.stop()


    # 4. Explicitly ensure all numerical columns are floats 
    # This is the final step to eliminate any chance of string/object types confusing the Imputer/Scaler
    input_df['city_development_index'] = input_df['city_development_index'].astype(float)
    input_df['training_hours'] = input_df['training_hours'].astype(float) 
    input_df['experience'] = input_df['experience'].astype(float) 

    
    with st.spinner('Calculating prediction...'):
        prediction_proba = model.predict_proba(input_df)[0][1]
        
    st.markdown("## Prediction Result")

    if prediction_proba >= 0.5:
        st.error("HIGH LIKELIHOOD OF CANDIDATE SEEKING JOB CHANGE")
        st.metric("Probability of Change (Target=1)", f"{prediction_proba * 100:.2f}%", delta_color="inverse")
    else:
        st.success("LOW LIKELIHOOD OF CANDIDATE SEEKING JOB CHANGE")
        st.metric("Probability of Change (Target=1)", f"{prediction_proba * 100:.2f}%")
        
    st.caption(f"Confidence score for the current inputs: {abs(prediction_proba - 0.5) * 200:.2f}% (measured from the 50% threshold)")
