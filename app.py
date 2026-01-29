import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Load files
# -----------------------------
model = joblib.load("student_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("feature_names.pkl")
r2 = joblib.load("r2_score.pkl")
mae = joblib.load("mae.pkl")

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("🧑‍🎓 Student Details")

gender = st.sidebar.selectbox("Gender", ["male", "female"])
race = st.sidebar.selectbox(
    "Race/Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)
parent_edu = st.sidebar.selectbox(
    "Parental Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ],
)
lunch = st.sidebar.selectbox("Lunch Type", ["standard", "free/reduced"])
prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])

reading = st.sidebar.slider("Reading Score", 0, 100, 50)
math = st.sidebar.slider("Math Score", 0, 100, 50)

# -----------------------------
# Predict button
# -----------------------------
if st.sidebar.button("🔮 Predict"):
    input_data = {
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent_edu,
        "lunch": lunch,
        "test preparation course": prep,
        "reading score": reading,
        "math score": math,
    }

    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=features, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Prediction
    prediction = model.predict(df_scaled)[0]

    # Grade logic
    if prediction >= 75:
        grade = "A 🟢"
    elif prediction >= 60:
        grade = "B 🟡"
    elif prediction >= 40:
        grade = "C 🟠"
    else:
        grade = "D 🔴"

    # -----------------------------
    # Main UI
    # -----------------------------
    st.markdown("## 🎓 Student Performance Prediction App")
    st.success(f"📊 **Predicted Score:** {prediction:.2f}")
    st.info(f"🏆 **Grade:** {grade}")

    col1, col2 = st.columns(2)
    col1.metric("R² Score", round(r2, 3))
    col2.metric("MAE", round(mae, 2))

    # -----------------------------
    # Graph
    # -----------------------------
    fig, ax = plt.subplots()
    ax.bar(
        ["Math Score", "Reading Score", "Predicted Writing Score"],
        [math, reading, prediction]
    )
    ax.set_ylabel("Score")
    ax.set_title("📈 Score Comparison")
    st.pyplot(fig)
