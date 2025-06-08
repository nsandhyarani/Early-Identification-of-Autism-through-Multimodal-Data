import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import warnings
warnings.filterwarnings("ignore")

# === Load Models & Preprocessing Tools ===
@st.cache_resource
def load_all():
    cnn_model = load_model("VGG16_best.h5")
    ann_model = load_model("autism_model.h5")
    scaler = joblib.load("scaler.pkl")
    encodings = joblib.load("t_int_encodings.pkl")
    return cnn_model, ann_model, scaler, encodings

cnn_model, ann_model, scaler, encodings = load_all()

# === Streamlit Layout ===
st.title("🧠 ASD Prediction from Image + Survey")
st.write("Upload a facial image and answer survey questions to get the final ASD prediction.")

# === Image Upload ===
img_file = st.file_uploader("Upload Facial Image", type=["jpg", "png", "jpeg"])
cnn_prob, cnn_class, cnn_asd_prob = None, None, None
if img_file:
    img = Image.open(img_file).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = keras_image.img_to_array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = cnn_model.predict(img_batch)
    cnn_prob = float(prediction[0][0])
    cnn_class = "NASD" if cnn_prob > 0.5 else "ASD"
    cnn_asd_prob = 1.0 - cnn_prob if cnn_class == "NASD" else cnn_prob

    st.image(img, caption=f"CNN Prediction: {cnn_class} ({cnn_prob:.2f})", width=300)

# === Survey Section ===
st.subheader("📝 Autism Behavior Survey")

autism_questions = {
    "A1_Score": "Does your child doesn't look at you when you call their name?",
    "A2_Score": "Does your child doesn't smile back at you when you smile at them?",
    "A3_Score": "Does your child engage in pretend play (e.g., talking on a toy phone)?",
    "A4_Score": "Does your child point at things they are interested in?",
    "A5_Score": "Does your child won't respond when spoken to?",
    "A6_Score": "Does your child doesn't make eye contact with people?",
    "A7_Score": "Does your child get upset by everyday noises (e.g., vacuum cleaner)?",
    "A8_Score": "Does your child repeat actions over and over again?",
    "A9_Score": "Does your child have difficulty with change in routine?",
    "A10_Score": "Does your child doesn't play with other children appropriately?"
}

qs = {key: st.selectbox(f"{question} (0 = No, 1 = Yes)", [0, 1], key=key)
       for key, question in autism_questions.items()}

age = st.number_input("Age", min_value=1, max_value=100, value=5)
gender = st.selectbox("Gender", list(encodings['gender'].keys()))
ethnicity = st.selectbox("Ethnicity", list(encodings['ethnicity'].keys()))
jundice = st.selectbox("Jaundice History", list(encodings['jundice'].keys()))
country = st.selectbox("Country", list(encodings['contry_of_res'].keys()))
used_app = st.selectbox("Used App Before", list(encodings['used_app_before'].keys()))
score = st.slider("Screening Score", 0, 20, 10)
relation = st.selectbox("Relation", list(encodings['relation'].keys()))

# === Prepare ANN Input ===
ann_input = {
    **qs,
    "age": age,
    "gender": gender,
    "ethnicity": ethnicity,
    "jundice": jundice,
    "contry_of_res": country,
    "used_app_before": used_app,
    "result": score,
    "relation": relation
}
ann_df = pd.DataFrame([ann_input])
categorical_cols = ['gender', 'ethnicity', 'jundice', 'contry_of_res', 'used_app_before', 'relation']
for col in categorical_cols:
    ann_df[col] = ann_df[col].map(encodings[col])
    ann_df[col] = ann_df[col].fillna(encodings[col].mean())

scaled_input = scaler.transform(ann_df)
ann_pred = ann_model.predict(scaled_input)
ann_prob = float(ann_pred[0][0])
ann_class = "ASD" if ann_prob > 0.5 else "NASD"
ann_asd_prob = 1.0 - ann_prob if ann_class == "NASD" else ann_prob

# === Display ANN Result ===
st.subheader("📊 ANN Survey Prediction")
st.info(f"Survey Model Prediction: {ann_class} (Probability: {ann_prob:.2f})")

# === Final Prediction ===
if cnn_prob is not None:
    final_prob = 0.6 * cnn_asd_prob + 0.4 * ann_asd_prob
    final_class = "ASD" if final_prob >= 0.5 else "No ASD"

    st.subheader("✅ Final Combined Prediction")
    st.success(f"**{final_class}** (Confidence Score: {final_prob:.2f})")

    # Matplotlib Annotated Image
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    ax.text(20, 40, f"CNN: {cnn_class} ({cnn_prob:.2f})", color="red", fontsize=14,
            bbox=dict(facecolor='yellow', alpha=0.7))
    ax.text(20, 70, f"ANN: {ann_class} ({ann_prob:.2f})", color="blue", fontsize=14,
            bbox=dict(facecolor='lightgreen', alpha=0.7))
    ax.text(20, 100, f"Final: {final_class} ({final_prob:.2f})", color="black", fontsize=14,
            bbox=dict(facecolor='lightblue', alpha=0.7))
    st.pyplot(fig)
