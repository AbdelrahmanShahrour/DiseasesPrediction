import streamlit as st 
import numpy as np
import pickle
import pandas as pd

st.set_page_config(
    page_title="Diseases Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# Diseases prediction app, abdalrahman shahrour",
    }
)

st.write('# Diseases Prediction App')

col1, col2, col3 = st.columns(3)

with col1:
    itching = st.radio(
        "itching",
        ('No', 'Yes'))

    skin_rash = st.radio(
        "skin rash",
        ('No', 'Yes'))

    nodal_skin_eruptions = st.radio(
        "nodal_skin_eruptions",
        ('No', 'Yes'))

    continuous_sneezing = st.radio(
        "continuous_sneezing",
        ('No', 'Yes'))

    shivering = st.radio(
        "shivering",
        ('No', 'Yes'))

    joint_pain = st.radio(
        "joint_pain",
        ('No', 'Yes'))

    acidity = st.radio(
        "acidity",
        ('No', 'Yes'))

    ulcers_on_tongue = st.radio(
        "ulcers_on_tongue",
        ('No', 'Yes'))

    muscle_wasting = st.radio(
        "muscle_wasting",
        ('No', 'Yes'))

    vomiting = st.radio(
        "vomiting",
        ('No', 'Yes'))

    spotting_urination = st.radio(
        "spotting urination",
        ('No', 'Yes'))

    fatigue = st.radio(
        "fatigue",
        ('No', 'Yes'))

    anxiety = st.radio(
        "anxiety",
        ('No', 'Yes'))

    mood_swings = st.radio(
        "mood_swings",
        ('No', 'Yes'))

    weight_loss = st.radio(
        "weight_loss",
        ('No', 'Yes'))

    lethargy = st.radio(
        "lethargy",
        ('No', 'Yes'))

    patches_in_throat = st.radio(
        "patches_in_throat",
        ('No', 'Yes'))

    irregular_sugar_level = st.radio(
        "irregular_sugar_level",
        ('No', 'Yes'))

    high_fever = st.radio(
        "high_fever",
        ('No', 'Yes'))

    sunken_eyes = st.radio(
        "sunken_eyes",
        ('No', 'Yes'))

    breathlessness = st.radio(
        "breathlessness",
        ('No', 'Yes'))

    sweating = st.radio(
        "sweating",
        ('No', 'Yes'))

    dehydration = st.radio(
        "dehydration",
        ('No', 'Yes'))

    indigestion = st.radio(
        "indigestion",
        ('No', 'Yes'))

    headache = st.radio(
        "headache",
        ('No', 'Yes'))

    dark_urine = st.radio(
        "dark_urine",
        ('No', 'Yes'))

    nausea = st.radio(
        "nausea",
        ('No', 'Yes'))

    loss_of_appetite = st.radio(
        "loss_of_appetite",
        ('No', 'Yes'))

    constipation = st.radio(
        "constipation",
        ('No', 'Yes'))

    abdominal_pain = st.radio(
        "abdominal_pain",
        ('No', 'Yes'))


    diarrhoea = st.radio(
        "diarrhoea",
        ('No', 'Yes'))

    mild_fever = st.radio(
        "mild_fever",
        ('No', 'Yes'))

    yellowing_of_eyes = st.radio(
        "yellowing_of_eyes",
        ('No', 'Yes'))

with col2:
    acute_liver_failure = st.radio(
        "acute_liver_failure",
        ('No', 'Yes'))

    swelling_of_stomach = st.radio(
        "swelling_of_stomach",
        ('No', 'Yes'))

    swelled_lymph_nodes = st.radio(
        "swelled_lymph_nodes",
        ('No', 'Yes'))

    malaise = st.radio(
        "malaise",
        ('No', 'Yes'))

    blurred_and_distorted_vision = st.radio(
        "blurred_and_distorted_vision",
        ('No', 'Yes'))

    phlegm = st.radio(
        "phlegm",
        ('No', 'Yes'))

    throat_irritation = st.radio(
        "throat_irritation",
        ('No', 'Yes'))

    redness_of_eyes = st.radio(
        "redness_of_eyes",
        ('No', 'Yes'))

    sinus_pressure = st.radio(
        "sinus_pressure",
        ('No', 'Yes'))

    runny_nose = st.radio(
        "runny_nose",
        ('No', 'Yes'))

    congestion = st.radio(
        "congestion",
        ('No', 'Yes'))

    chest_pain = st.radio(
        "chest_pain",
        ('No', 'Yes'))


    weakness_in_limbs = st.radio(
        "weakness_in_limbs",
        ('No', 'Yes'))

    fast_heart_rate = st.radio(
        "fast_heart_rate",
        ('No', 'Yes'))

    pain_during_bowel_movements = st.radio(
        "pain_during_bowel_movements",
        ('No', 'Yes'))

    pain_in_anal_region = st.radio(
        "pain_in_anal_region",
        ('No', 'Yes'))

    bloody_stool = st.radio(
        "bloody_stool",
        ('No', 'Yes'))

    irritation_in_anus = st.radio(
        "irritation_in_anus",
        ('No', 'Yes'))

    neck_pain = st.radio(
        "neck_pain",
        ('No', 'Yes'))

    dizziness = st.radio(
        "dizziness",
        ('No', 'Yes'))

    cramps = st.radio(
        "cramps",
        ('No', 'Yes'))

    bruising = st.radio(
        "bruising",
        ('No', 'Yes'))

    swollen_legs = st.radio(
        "swollen_legs",
        ('No', 'Yes'))

    swollen_blood_vessels = st.radio(
        "swollen_blood_vessels",
        ('No', 'Yes'))

    excessive_hunger = st.radio(
        "excessive_hunger",
        ('No', 'Yes'))

    extra_marital_contacts = st.radio(
        "extra_marital_contacts",
        ('No', 'Yes'))

    drying_and_tingling_lips = st.radio(
        "drying_and_tingling_lips",
        ('No', 'Yes'))

    slurred_speech = st.radio(
        "slurred_speech",
        ('No', 'Yes'))

    knee_pain = st.radio(
        "knee_pain",
        ('No', 'Yes'))

    hip_joint_pain = st.radio(
        "hip_joint_pain",
        ('No', 'Yes'))

    muscle_weakness = st.radio(
        "muscle_weakness",
        ('No', 'Yes'))

    stiff_neck = st.radio(
        "stiff_neck",
        ('No', 'Yes'))

    spinning_movements = st.radio(
        "spinning_movements",
        ('No', 'Yes'))

    

with col3:

    loss_of_balance = st.radio(
        "loss_of_balance",
        ('No', 'Yes'))

    unsteadiness = st.radio(
        "unsteadiness",
        ('No', 'Yes'))

    weakness_of_one_body_side = st.radio(
        "weakness_of_one_body_side",
        ('No', 'Yes'))

    loss_of_smell = st.radio(
        "loss_of_smell",
        ('No', 'Yes'))

    depression = st.radio(
        "depression",
        ('No', 'Yes'))

    irritability = st.radio(
        "irritability",
        ('No', 'Yes'))

    muscle_pain = st.radio(
        "muscle_pain",
        ('No', 'Yes'))


    altered_sensorium = st.radio(
        "altered_sensorium",
        ('No', 'Yes'))

    red_spots_over_body = st.radio(
        "red_spots_over_body",
        ('No', 'Yes'))

    abnormal_menstruation = st.radio(
        "abnormal_menstruation",
        ('No', 'Yes'))

    dischromic_patches = st.radio(
        "dischromic patches",
        ('No', 'Yes'))

    watering_from_eyes = st.radio(
        "watering_from_eyes",
        ('No', 'Yes'))

    increased_appetite = st.radio(
        "increased_appetite",
        ('No', 'Yes'))

    polyuria = st.radio(
        "polyuria",
        ('No', 'Yes'))

    family_history = st.radio(
        "family_history",
        ('No', 'Yes'))

    mucoid_sputum = st.radio(
        "mucoid_sputum",
        ('No', 'Yes'))

    rusty_sputum = st.radio(
        "rusty_sputum",
        ('No', 'Yes'))

    lack_of_concentration = st.radio(
        "lack_of_concentration",
        ('No', 'Yes'))

    visual_disturbances = st.radio(
        "visual_disturbances",
        ('No', 'Yes'))

    coma = st.radio(
        "coma",
        ('No', 'Yes'))

    stomach_bleeding = st.radio(
        "stomach_bleeding",
        ('No', 'Yes'))

    distention_of_abdomen = st.radio(
        "distention_of_abdomen",
        ('No', 'Yes'))

    history_of_alcohol_consumption = st.radio(
        "history_of_alcohol_consumption",
        ('No', 'Yes'))

    fluid_overload = st.radio(
        "fluid_overload.1",
        ('No', 'Yes'))

    prominent_veins_on_calf = st.radio(
        "prominent_veins_on_calf",
        ('No', 'Yes'))

    palpitations = st.radio(
        "palpitations",
        ('No', 'Yes'))

    pus_filled_pimples = st.radio(
        "pus_filled_pimples",
        ('No', 'Yes'))

    blackheads = st.radio(
        "blackheads",
        ('No', 'Yes'))

    scurring = st.radio(
        "scurring",
        ('No', 'Yes'))

    skin_peeling = st.radio(
        "skin_peeling",
        ('No', 'Yes'))

    silver_like_dusting = st.radio(
        "silver_like_dusting",
        ('No', 'Yes'))

    small_dents_in_nails = st.radio(
        "small_dents_in_nails",
        ('No', 'Yes'))

    inflammatory_nails = st.radio(
        "inflammatory_nails",
        ('No', 'Yes'))


def get_res(ans:str) -> int:
    if ans == "Yes":
        return 1
    else:
        return 0

data = [
    get_res(itching),
 get_res(skin_rash),
 get_res(nodal_skin_eruptions),
 get_res(continuous_sneezing),
 get_res(shivering),
 get_res(joint_pain),
 get_res(acidity),
 get_res(ulcers_on_tongue),
 get_res(muscle_wasting),
 get_res(vomiting),
 get_res(spotting_urination),
 get_res(fatigue),
 get_res(anxiety),
 get_res(mood_swings),
 get_res(weight_loss),
 get_res(lethargy),
 get_res(patches_in_throat),
 get_res(irregular_sugar_level),
 get_res(high_fever),
 get_res(sunken_eyes),
 get_res(breathlessness),
 get_res(sweating),
 get_res(dehydration),
 get_res(indigestion),
 get_res(headache),
 get_res(dark_urine),
 get_res(nausea),
 get_res(loss_of_appetite),
 get_res(constipation),
 get_res(abdominal_pain),
 get_res(diarrhoea),
 get_res(mild_fever),
 get_res(yellowing_of_eyes),
 get_res(acute_liver_failure),
 get_res(swelling_of_stomach),
 get_res(swelled_lymph_nodes),
 get_res(malaise),
 get_res(blurred_and_distorted_vision),
 get_res(phlegm),
 get_res(throat_irritation),
 get_res(redness_of_eyes),
 get_res(sinus_pressure),
 get_res(runny_nose),
 get_res(congestion),
 get_res(chest_pain),
 get_res(weakness_in_limbs),
 get_res(fast_heart_rate),
 get_res(pain_during_bowel_movements),
 get_res(pain_in_anal_region),
 get_res(bloody_stool),
 get_res(irritation_in_anus),
 get_res(neck_pain),
 get_res(dizziness),
 get_res(cramps),
 get_res(bruising),
 get_res(swollen_legs),
 get_res(swollen_blood_vessels),
 get_res(excessive_hunger),
 get_res(extra_marital_contacts),
 get_res(drying_and_tingling_lips),
 get_res(slurred_speech),
 get_res(knee_pain),
 get_res(hip_joint_pain),
 get_res(muscle_weakness),
 get_res(stiff_neck),
 get_res(spinning_movements),
 get_res(loss_of_balance),
 get_res(unsteadiness),
 get_res(weakness_of_one_body_side),
 get_res(loss_of_smell),
 get_res(depression),
 get_res(irritability),
 get_res(muscle_pain),
 get_res(altered_sensorium),
 get_res(red_spots_over_body),
 get_res(abnormal_menstruation),
 get_res(dischromic_patches),
 get_res(watering_from_eyes),
 get_res(increased_appetite),
 get_res(polyuria),
 get_res(family_history),
 get_res(mucoid_sputum),
 get_res(rusty_sputum),
 get_res(lack_of_concentration),
 get_res(visual_disturbances),
 get_res(coma),
 get_res(stomach_bleeding),
 get_res(distention_of_abdomen),
 get_res(history_of_alcohol_consumption),
 get_res(fluid_overload),
 get_res(prominent_veins_on_calf),
 get_res(palpitations),
 get_res(pus_filled_pimples),
 get_res(blackheads),
 get_res(scurring),
 get_res(skin_peeling),
 get_res(silver_like_dusting),
 get_res(small_dents_in_nails),
 get_res(inflammatory_nails)
]


def get_predict(data:list) -> np.ndarray:
    data = np.array(data).reshape(-1, 1)
    data = pd.DataFrame(data).T
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    return loaded_model.predict(data)

def decoder(encoder:np.ndarray) -> str:
    prognosis = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']
    return prognosis[int(encoder)]

Pred = decoder(get_predict(data))

st.write(f'## My Prediction : {Pred}')