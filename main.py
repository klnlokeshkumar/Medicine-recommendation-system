# import modules
import streamlit as st
import numpy as np
import pandas as pd
import pickle 

# load datasets
precautions_df = pd.read_csv("datasets/precautions_df.csv")
workout_df = pd.read_csv("datasets/workout_df.csv")
description_df = pd.read_csv("datasets/description.csv")
medications_df = pd.read_csv("datasets/medications.csv")
diets_df = pd.read_csv("datasets/diets.csv")

# load model from pickle file
from pathlib import Path

model_path = Path("./assets/svc.pkl")
if model_path.exists():
    svc = pickle.load(open(model_path, "rb"))
else:
    st.error(f"Model file not found at {model_path}")
    st.stop()

# predict disease using the user input
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

#get the details of the disease like description preacutions etc
def helper(disease):
    # get description from description_df
    desc = description_df[description_df['Disease'] == disease]['Description']
    desc = " ".join (w for w in desc)
    # get precautions from precaution_df
    prec = precautions_df[precautions_df['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    prec = [w for w in prec.values]
    # get medication from medications_df
    medi = medications_df[medications_df['Disease'] == disease]['Medication']
    medi = [w for w in medi.values]
    # get diets from diets_df
    diet = diets_df[diets_df['Disease'] == disease]['Diet']
    diet = [w for w in diet.values]
    # get workouts from workouts_df
    work = workout_df[workout_df['disease'] == disease]['workout'].tolist()

    return desc, prec, medi, diet, work

# web app streamlit 
st.set_page_config("Medicine Recommendation", "🩺", "wide")

st.markdown(
    """
     <style>
    body {
        background-color: #D6EAF8;
    }
    .stApp {
        background-color: #D6EAF8;
    }
    .css-18e3th9 {
        padding: 0px !important;
    }
    .css-1v3fvcr {
        padding: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True
)

# adjust the ratio as needed
col1, col2 = st.columns([1, 3])
with col1:
    st.image("./assets/image.png")
with col2:
    st.markdown("""
                <div style='display: flex; background-color: #3498DB; color: #2C3E50; justify-content: center; align-items: center; height: 100%;'>
                    <h1>Health Care Center</h1>
                </div>""", unsafe_allow_html=True)
    
symptoms = st.multiselect("**Symptoms", symptoms_dict)

if st.button("Predict", key="Predict_button"):
    if not symptoms:
        st.error("Please enter the symptoms.")
    elif any(char.isdigit() for char in symptoms):
        st.error("Symptoms should only contain alphanumeric characters.")
    else:
        # patient_symptoms = input("Enter the symptoms....")
        patient_symptoms = [s.strip() for s in symptoms]
        patient_symptoms = [s.strip("[]' ") for s in patient_symptoms]
        predicted_disease = get_predicted_value(patient_symptoms)
        desc, prec, medi, diet, work = helper(predicted_disease)

        st.markdown(f"""
        <div style="background-color: #f4fbfc; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); color: #000; font-family: Arial, sans-serif;">
            <h3 style="margin-bottom: 10px; text-align: center; color: #000;">Predicted Disease</h3>
            <p style="font-size: 18px; font-weight: bold; text-align: center; color: #555;">{predicted_disease}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background-color: #e8f6f9; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); color: #000; font-family: Arial, sans-serif;">
            <h4 style="margin-bottom: 10px; color: #000;">Description</h4>
            <p style="color: #555;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #e8f6f9; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); color: #000; font-family: Arial, sans-serif;">
            <h4 style="margin-bottom: 10px; color: #000;">Precautions</h4>
            <ul style="margin-left: 20px;">
        """, unsafe_allow_html=True)
        for pre in prec[0]:
            st.markdown(f"<li style='color: #555;'>{pre}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #e8f6f9; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); color: #000; font-family: Arial, sans-serif;">
            <h4 style="margin-bottom: 10px; color: #000;">Medications</h4>
            <ul style="margin-left: 20px;">
        """, unsafe_allow_html=True)
        for med in medi:
            st.markdown(f"<li style='color: #555;'>{med}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #e8f6f9; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); color: #000; font-family: Arial, sans-serif;">
            <h4 style="margin-bottom: 10px; color: #000;">Recommended Diet</h4>
            <ul style="margin-left: 20px;">
        """, unsafe_allow_html=True)
        for diet in diet:
            st.markdown(f"<li style='color: #555;'>{diet}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background-color: #e8f6f9; padding: 15px; border-radius: 10px; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); color: #000; font-family: Arial, sans-serif;">
            <h4 style="margin-bottom: 10px; color: #000;">Workout Suggestions</h4>
            <ul style="margin-left: 20px;">
        """, unsafe_allow_html=True)
        for wrk in work:
            st.markdown(f"<li style='color: #555;'>{wrk}</li>", unsafe_allow_html=True)
        st.markdown("</ul></div>", unsafe_allow_html=True)
