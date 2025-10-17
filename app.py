import streamlit as st
import pandas as pd
from datetime import datetime
from dga_logic import classify_fault
import os
import tempfile
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- Safe cross-platform paths ---
TMP_DIR = tempfile.gettempdir()
DATA_DIR = os.path.join(TMP_DIR, "dga_data")
MODEL_DIR = os.path.join(TMP_DIR, "dga_models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "dga_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# --- Initialize CSV if missing ---
if not os.path.exists(DATA_PATH):
    df_init = pd.DataFrame(
        columns=[
            "Timestamp", "H2", "CH4", "C2H2", "C2H4", "C2H6",
            "CO", "CO2", "RuleBasedFault", "ExpertLabel"
        ]
    )
    df_init.to_csv(DATA_PATH, index=False)

# --- Streamlit setup ---
st.set_page_config(page_title="DGA AI Training Camp", layout="wide")
st.title("🧠 DGA AI Training Camp v3.5")
st.caption("Collect, train, upload and test transformer fault analysis AI models.")

# ======================================
# SECTION 1: Data Entry & Fault Logic
# ======================================
st.header("📥 Data Entry & Rule-Based Fault Classification")

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
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception:
            df = pd.DataFrame(columns=[
                "Timestamp", "H2", "CH4", "C2H2", "C2H4", "C2H6",
                "CO", "CO2", "RuleBasedFault", "ExpertLabel"
            ])

        new_row = {
            "Timestamp": timestamp, "H2": h2, "CH4": ch4, "C2H2": c2h2,
            "C2H4": c2h4, "C2H6": c2h6, "CO": co, "CO2": co2,
            "RuleBasedFault": fault, "ExpertLabel": expert_label if expert_label else fault
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        st.success(f"✅ Entry saved for AI training! ({timestamp})")

# ======================================
# SECTION 2: Search & Filter
# ======================================
st.header("🔍 Search & Filter Saved Data")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    search_term = st.text_input("Search (fault type, label, or date)", placeholder="e.g. T2, D1, 2025-10-17")
    fault_filter = st.selectbox("Filter by Fault Type", ["All"] + sorted(df["RuleBasedFault"].dropna().unique().tolist()))
    filtered_df = df.copy()

    if search_term:
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        ]
    if fault_filter != "All":
        filtered_df = filtered_df[filtered_df["RuleBasedFault"] == fault_filter]

    st.dataframe(filtered_df)
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data", csv, "filtered_training_data.csv", "text/csv")

else:
    st.warning("⚠️ No data found. Add entries above first.")

# ======================================
# SECTION 3: External Dataset Upload
# ======================================
st.header("📤 Upload External DGA Dataset")

st.markdown("""
Upload a `.csv` file containing DGA gas readings and fault labels.  
It should include the following columns:
**H2, CH4, C2H2, C2H4, C2H6, CO, CO2, ExpertLabel**
""")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! {df_upload.shape[0]} rows detected.")
        st.dataframe(df_upload.head())

        required_cols = {"H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2", "ExpertLabel"}
        if required_cols.issubset(df_upload.columns):
            if st.button("🔄 Merge with Training Data"):
                df_existing = pd.read_csv(DATA_PATH)
                df_upload["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df_upload["RuleBasedFault"] = df_upload["ExpertLabel"]
                df_combined = pd.concat([df_existing, df_upload], ignore_index=True)
                df_combined.to_csv(DATA_PATH, index=False)
                st.success(f"✅ Dataset merged! New total: {df_combined.shape[0]} entries.")
        else:
            st.error(f"❌ Missing columns! Required: {', '.join(required_cols)}")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# ======================================
# SECTION 4: AI Model Training
# ======================================
st.header("🤖 Train AI Model")

if st.button("Train Model"):
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if len(df) < 10:
            st.warning("⚠️ Not enough data to train (need at least 10 samples).")
        else:
            df = df.dropna(subset=["ExpertLabel"])
            X = df[["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]]
            y = df["ExpertLabel"]

            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"✅ Model trained! Accuracy: {acc*100:.2f}%")
            st.code(classification_report(y_test, y_pred))
            st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

            joblib.dump(model, MODEL_PATH)
            joblib.dump(encoder, ENCODER_PATH)
            st.success("💾 Model saved successfully.")
    else:
        st.error("No training data available.")

# ======================================
# SECTION 5: AI Prediction Test
# ======================================
st.header("🔮 Test AI Model Predictions")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    col1, col2, col3 = st.columns(3)
    with col1:
        test_h2 = st.number_input("H₂ (ppm) test", min_value=0.0, value=80.0)
        test_ch4 = st.number_input("CH₄ (ppm) test", min_value=0.0, value=120.0)
        test_c2h2 = st.number_input("C₂H₂ (ppm) test", min_value=0.0, value=10.0)
    with col2:
        test_c2h4 = st.number_input("C₂H₄ (ppm) test", min_value=0.0, value=45.0)
        test_c2h6 = st.number_input("C₂H₆ (ppm) test", min_value=0.0, value=60.0)
        test_co = st.number_input("CO (ppm) test", min_value=0.0, value=750.0)
    with col3:
        test_co2 = st.number_input("CO₂ (ppm) test", min_value=0.0, value=4200.0)

    if st.button("Run AI Prediction"):
        sample = [[test_h2, test_ch4, test_c2h2, test_c2h4, test_c2h6, test_co, test_co2]]
        pred = model.predict(sample)[0]
        fault_label = encoder.inverse_transform([pred])[0]
        st.subheader(f"🔮 AI Predicted Fault: {fault_label}")
else:
    st.info("⚙️ Train a model first before using this feature.")

st.markdown("---")
st.caption("Developed by Code GPT 🧑‍💻 | DGA Analysis AI System v3.5 | Supports Global Dataset Uploads 🌐")
