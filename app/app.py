
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Wellness Tourism Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Wellness Tourism Package Predictor")

st.write(
    "Enter customer details to estimate the probability "
    "of purchasing the Wellness Tourism Package."
)

model_path = Path(__file__).resolve().parent.parent / "models" / "tourism_purchase_model.pkl"
model = joblib.load(model_path)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

type_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

duration_pitch = st.number_input(
    "Duration of Pitch",
    min_value=1.0,
    max_value=60.0,
    value=15.0
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Freelancer", "Small Business", "Large Business"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

persons = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=20,
    value=2
)

followups = st.number_input(
    "Number of Follow-ups",
    min_value=0,
    max_value=10,
    value=3
)

product = st.selectbox(
    "Product Pitched",
    ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"]
)

property_star = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

trips = st.number_input(
    "Number of Trips",
    min_value=0.0,
    max_value=30.0,
    value=3.0
)

passport = st.selectbox(
    "Passport",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

pitch_score = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

own_car = st.selectbox(
    "Own Car",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

children = st.number_input(
    "Number of Children Visiting",
    min_value=0.0,
    max_value=10.0,
    value=0.0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

income = st.number_input(
    "Monthly Income",
    min_value=0.0,
    max_value=1000000.0,
    value=20000.0
)

if st.button("Predict Purchase Probability"):

    input_data = pd.DataFrame({
        "Age": [age],
        "TypeofContact": [type_contact],
        "CityTier": [city_tier],
        "DurationOfPitch": [duration_pitch],
        "Occupation": [occupation],
        "Gender": [gender],
        "NumberOfPersonVisiting": [persons],
        "NumberOfFollowups": [followups],
        "ProductPitched": [product],
        "PreferredPropertyStar": [property_star],
        "MaritalStatus": [marital_status],
        "NumberOfTrips": [trips],
        "Passport": [passport],
        "PitchSatisfactionScore": [pitch_score],
        "OwnCar": [own_car],
        "NumberOfChildrenVisiting": [children],
        "Designation": [designation],
        "MonthlyIncome": [income]
    })

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = model.predict(
        input_data
    )[0]

    st.subheader("Prediction Result")

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.success(
            "High potential customer — consider prioritizing this customer."
        )
    else:
        st.info(
            "Lower purchase probability — consider lower-priority targeting."
        )
