import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Load model and selected features
# -----------------------------------
model = joblib.load("tuned_decision_tree.pkl")
selected_features = joblib.load("selected_features.pkl")

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(
    page_title="Skin Cancer Risk Checker",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------------
# Title & intro
# -----------------------------------
st.title("🩺 Skin Cancer Risk Checker")

st.write("""
This tool provides an **early risk screening** for skin cancer based on
health and lifestyle information.

Please answer the questions below as accurately as possible.
""")

st.info("⚠️ This tool is for educational purposes only and is **not** a medical diagnosis.")

st.divider()

# -----------------------------------
# User input section
# -----------------------------------
st.header("👤 Personal & Health Information")

user_input = {}

# -----------------------------------
# Age (USER-FRIENDLY)
# -----------------------------------
age_groups = [
    "18–24", "25–29", "30–34", "35–39", "40–44", "45–49",
    "50–54", "55–59", "60–64", "65–69", "70–74", "75–79", "80+"
]

age_mapping = {label: idx for idx, label in enumerate(age_groups)}

age_selected = st.selectbox(
    "Age Group",
    age_groups,
    help="Select the age range that best describes you"
)

user_input["age_category"] = age_mapping[age_selected]

# -----------------------------------
# Sex
# -----------------------------------
sex = st.radio("Sex", ["Female", "Male"])
user_input["sex_Male"] = 1 if sex == "Male" else 0

# -----------------------------------
# Body measurements
# -----------------------------------
st.subheader("📏 Body Measurements")

user_input["height_cm"] = st.slider(
    "Height (cm)",
    140, 200, 170
)

user_input["weight_kg"] = st.slider(
    "Weight (kg)",
    40, 150, 70
)

user_input["bmi"] = st.slider(
    "Body Mass Index (BMI)",
    15.0, 40.0, 23.0,
    help="BMI is a measure of body fat based on height and weight"
)

# -----------------------------------
# Lifestyle habits
# -----------------------------------
st.subheader("🥗 Lifestyle & Diet")

user_input["fruit_consumption"] = st.slider(
    "Fruit consumption (servings per week)",
    0, 50, 7
)

user_input["green_vegetables_consumption"] = st.slider(
    "Green vegetable consumption (servings per week)",
    0, 50, 7
)

user_input["friedpotato_consumption"] = st.slider(
    "Fried food consumption (times per week)",
    0, 20, 2
)

user_input["alcohol_consumption"] = st.slider(
    "Alcohol consumption (drinks per week)",
    0, 30, 0
)

# -----------------------------------
# Medical history
# -----------------------------------
st.subheader("🩺 Medical History")

user_input["general_health"] = st.selectbox(
    "General Health",
    [0, 1, 2, 3, 4],
    format_func=lambda x: ["Poor", "Fair", "Good", "Very Good", "Excellent"][x]
)

user_input["smoking_history"] = st.selectbox(
    "Smoking History",
    [0, 1, 2, 3],
    format_func=lambda x: ["Never", "Former", "Occasional", "Current"][x]
)

user_input["checkup"] = st.radio(
    "Have you had a routine medical checkup in the past year?",
    ["No", "Yes"]
)
user_input["checkup"] = 1 if user_input["checkup"] == "Yes" else 0

user_input["other_cancer"] = st.radio(
    "Have you ever been diagnosed with another type of cancer?",
    ["No", "Yes"]
)
user_input["other_cancer"] = 1 if user_input["other_cancer"] == "Yes" else 0

user_input["heart_disease"] = st.radio(
    "Heart Disease",
    ["No", "Yes"]
)
user_input["heart_disease"] = 1 if user_input["heart_disease"] == "Yes" else 0

user_input["diabetes_Yes"] = st.radio(
    "Diabetes",
    ["No", "Yes"]
)
user_input["diabetes_Yes"] = 1 if user_input["diabetes_Yes"] == "Yes" else 0

user_input["depression"] = st.radio(
    "History of Depression",
    ["No", "Yes"]
)
user_input["depression"] = 1 if user_input["depression"] == "Yes" else 0

user_input["arthritis"] = st.radio(
    "Arthritis",
    ["No", "Yes"]
)
user_input["arthritis"] = 1 if user_input["arthritis"] == "Yes" else 0

user_input["exercise"] = st.radio(
    "Do you exercise regularly?",
    ["No", "Yes"]
)
user_input["exercise"] = 1 if user_input["exercise"] == "Yes" else 0

# -----------------------------------
# Prediction
# -----------------------------------
st.divider()
st.header("📊 Prediction Result")

input_df = pd.DataFrame([user_input])[selected_features]

if st.button("🔍 Check Skin Cancer Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("⚠️ Higher Risk Detected")
        st.write(f"The model estimates a **{probability:.1%} risk** of skin cancer.")
        st.write("Please consider consulting a medical professional.")
    else:
        st.success("✅ Lower Risk Detected")
        st.write(f"The model estimates a **{probability:.1%} risk** of skin cancer.")
        st.write("Continue maintaining healthy lifestyle habits.")

# -----------------------------------
# Footer
# -----------------------------------
st.caption(
    "This application uses a tuned Decision Tree model trained on population health data. "
    "Results should not be used as a substitute for professional medical advice."
)
