import streamlit as st
import pandas as pd
from datetime import datetime
from dga_logic import classify_fault
import os
import tempfile
import joblib
import re
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# =====================================================
# DGA AI TRAINING CAMP v4.3 — Analytics Edition
# =====================================================

TMP_DIR = tempfile.gettempdir()
DATA_DIR = os.path.join(TMP_DIR, "dga_data")
MODEL_DIR = os.path.join(TMP_DIR, "dga_models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "dga_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Initialize dataset if missing
if not os.path.exists(DATA_PATH):
    df_init = pd.DataFrame(columns=[
        "Timestamp", "H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2",
        "RuleBasedFault", "ExpertLabel"
    ])
    df_init.to_csv(DATA_PATH, index=False)

# --- Streamlit setup ---
st.set_page_config(page_title="DGA AI Training Camp v4.3", layout="wide")
st.title("🧠 DGA AI Training Camp v4.3 — Analytics Edition")
st.caption("Advanced Dissolved Gas Analysis system with full AI training, visual analytics, and fault classification.")

# ======================================
# SECTION 1: Data Entry
# ======================================
st.header("📥 Manual Data Entry & Rule-Based Classification")

col1, col2, col3 = st.columns(3)
with col1:
    h2 = st.number_input("H₂ (ppm)", min_value=0.0, value=50.0)
    ch4 = st.number_input("CH₄ (ppm)", min_value=0.0, value=100.0)
    c2h2 = st.number_input("C₂H₂ (ppm)", min_value=0.0, value=5.0)
with col2:
    c2h4 = st.number_input("C₂H₄ (ppm)", min_value=0.0, value=40.0)
    c2h6 = st.number_input("C₂H₆ (ppm)", min_value=0.0, value=50.0)
    co = st.number_input("CO (ppm)", min_value=0.0, value=800.0)
with col3:
    co2 = st.number_input("CO₂ (ppm)", min_value=0.0, value=4000.0)

if st.button("Run Rule-Based Analysis"):
    fault, explanation = classify_fault(h2, ch4, c2h2, c2h4, c2h6, co, co2)
    st.subheader(f"🧾 Fault Type: {fault}")
    st.caption(explanation)

    expert_label = st.text_input("Expert Label (optional)", placeholder="Enter confirmed fault if different")
    if st.button("💾 Save Entry"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.read_csv(DATA_PATH)
        new_row = {
            "Timestamp": timestamp, "H2": h2, "CH4": ch4, "C2H2": c2h2,
            "C2H4": c2h4, "C2H6": c2h6, "CO": co, "CO2": co2,
            "RuleBasedFault": fault, "ExpertLabel": expert_label if expert_label else fault
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        st.success(f"✅ Entry saved for AI training! ({timestamp})")

# ======================================
# SECTION 2: Smart Upload + Cleaning + Visualization
# ======================================
st.header("📤 Upload DGA Dataset (Multi-Phase + Visualization + Analytics)")

def normalize_column_names(columns):
    normalized = []
    for c in columns:
        c_lower = c.lower().strip()
        c_lower = re.sub(r'^(red|yellow|blue)[:\\s-]+', '', c_lower)
        c_lower = c_lower.replace('(ppm)', '').strip()
        if re.search(r'h.?ydrogen', c_lower): normalized.append('H2')
        elif re.search(r'meth.?ane', c_lower): normalized.append('CH4')
        elif re.search(r'acetyl', c_lower): normalized.append('C2H2')
        elif re.search(r'ethyl.?ene', c_lower): normalized.append('C2H4')
        elif re.search(r'ethane', c_lower): normalized.append('C2H6')
        elif re.search(r'co2', c_lower): normalized.append('CO2')
        elif re.match(r'co(?!2)', c_lower): normalized.append('CO')
        elif re.search(r'date|time', c_lower): normalized.append('Timestamp')
        else: normalized.append(c.strip())
    return normalized

def average_duplicate_columns(df):
    """Smart averaging for duplicate gas columns (handles CO, CO2, etc.)."""
    averaged = pd.DataFrame()
    seen = set()

    for col in df.columns:
        if col in seen:
            continue
        duplicates = [c for c in df.columns if c == col]
        seen.update(duplicates)

        try:
            numeric = df[duplicates].apply(pd.to_numeric, errors="coerce")
            averaged[col] = numeric.mean(axis=1, skipna=True)
        except Exception:
            averaged[col] = df[duplicates[0]]
    return averaged

uploaded_file = st.file_uploader("📂 Choose a DGA dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        df_upload.columns = normalize_column_names(df_upload.columns)
        df_upload = average_duplicate_columns(df_upload)

        st.success(f"✅ File uploaded successfully! {df_upload.shape[0]} rows detected.")
        st.dataframe(df_upload.head())

        # Identify timestamp
        if "Timestamp" in df_upload.columns:
            df_upload["Timestamp"] = pd.to_datetime(df_upload["Timestamp"], errors="coerce")

        gases = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]
        df_upload = df_upload[[col for col in df_upload.columns if col in gases or col == "Timestamp"]].dropna(subset=gases)

        # Rule-based classification
        st.write("🔍 Applying rule-based fault classification...")
        df_upload["ExpertLabel"] = df_upload.apply(
            lambda row: classify_fault(
                row.get("H2", 0), row.get("CH4", 0),
                row.get("C2H2", 0), row.get("C2H4", 0),
                row.get("C2H6", 0), row.get("CO", 0), row.get("CO2", 0)
            )[0], axis=1
        )

        # --- Summary ---
        st.markdown("### 📊 Data Quality Summary")
        st.write("Rows:", df_upload.shape[0])
        st.write("Missing values per column:")
        st.write(df_upload.isna().sum())
        st.write("Summary statistics:")
        st.write(df_upload.describe())

        # --- Visualization ---
        st.markdown("### 📈 Time-Series Visualization")
        if "Timestamp" in df_upload.columns:
            gas_choice = st.multiselect("Select gases to visualize", gases, default=["H2", "CH4"])
            for gas in gas_choice:
                fig = px.line(df_upload, x="Timestamp", y=gas, title=f"{gas} Concentration Over Time", markers=True)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🥧 Fault Distribution")
        fig_fault = px.pie(df_upload, names="ExpertLabel", title="Fault Type Distribution")
        st.plotly_chart(fig_fault, use_container_width=True)

        # --- Correlation Heatmap ---
        st.markdown("### 🔥 Gas Correlation Heatmap")
        corr = df_upload[gases].corr()
        fig_heat = ff.create_annotated_heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="Viridis",
            showscale=True
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # --- Merge Dataset ---
        if st.button("🔄 Merge with Training Data"):
            df_existing = pd.read_csv(DATA_PATH)
            df_upload["RuleBasedFault"] = df_upload["ExpertLabel"]
            df_combined = pd.concat([df_existing, df_upload], ignore_index=True)
            df_combined.to_csv(DATA_PATH, index=False)
            st.success(f"✅ Dataset merged successfully! Total entries: {df_combined.shape[0]}")
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# ======================================
# SECTION 3: Train AI Model
# ======================================
st.header("🤖 Train AI Model")

if st.button("Train Model"):
    df = pd.read_csv(DATA_PATH)
    if len(df) < 10:
        st.warning("⚠️ Not enough data to train.")
    else:
        X = df[["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]]
        y = df["ExpertLabel"]
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")
        st.code(classification_report(y_test, y_pred))
        st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        st.info("💾 Model saved successfully!")

# ======================================
# SECTION 4: Test Model
# ======================================
st.header("🔮 Test AI Model")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    col1, col2, col3 = st.columns(3)
    with col1:
        test_h2 = st.number_input("H₂ (ppm)", min_value=0.0, value=80.0)
        test_ch4 = st.number_input("CH₄ (ppm)", min_value=0.0, value=120.0)
        test_c2h2 = st.number_input("C₂H₂ (ppm)", min_value=0.0, value=10.0)
    with col2:
        test_c2h4 = st.number_input("C₂H₄ (ppm)", min_value=0.0, value=45.0)
        test_c2h6 = st.number_input("C₂H₆ (ppm)", min_value=0.0, value=60.0)
        test_co = st.number_input("CO (ppm)", min_value=0.0, value=750.0)
    with col3:
        test_co2 = st.number_input("CO₂ (ppm)", min_value=0.0, value=4200.0)

    if st.button("Run AI Prediction"):
        sample = [[test_h2, test_ch4, test_c2h2, test_c2h4, test_c2h6, test_co, test_co2]]
        pred = model.predict(sample)[0]
        fault_label = encoder.inverse_transform([pred])[0]
        st.subheader(f"🔮 AI Predicted Fault: {fault_label}")
else:
    st.info("No trained model found. Train one first.")

st.markdown("---")
st.caption("Developed by Code GPT 🧑‍💻 | DGA AI System v4.3 | Analytics Edition 🌐")
