import streamlit as st
import pandas as pd
from datetime import datetime
from dga_logic import classify_fault
import os
import tempfile
import joblib
import re
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LinearRegression

# =====================================================
# 🧠 DGA AI TRAINING CAMP v4.7 — Dataset Manager + Anomaly Detection
# =====================================================

TMP_DIR = tempfile.gettempdir()
DATA_DIR = os.path.join(TMP_DIR, "dga_data")
MODEL_DIR = os.path.join(TMP_DIR, "dga_models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "dga_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

if not os.path.exists(DATA_PATH):
    df_init = pd.DataFrame(columns=[
        "Timestamp", "Transformer", "H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2",
        "RuleBasedFault", "ExpertLabel"
    ])
    df_init.to_csv(DATA_PATH, index=False)

st.set_page_config(page_title="DGA AI v4.7", layout="wide")
st.title("🧠 DGA AI Training Camp v4.7 — Dataset Manager + Anomaly Detection")
st.caption("Manage datasets, train AI, visualize timelines, and forecast transformer health.")

# =====================================================
# ⚙️ Utility Functions
# =====================================================
def normalize_column_names(columns):
    normalized = []
    for c in columns:
        c_lower = c.lower().strip()
        c_lower = re.sub(r'\[.*?\]|\(.*?\)|ppm|parts per million', '', c_lower)
        c_lower = re.sub(r'^(red|yellow|blue)[:\s-]+', '', c_lower)
        c_lower = re.sub(r'[\s:;_-]+', ' ', c_lower).strip()
        if re.search(r'h.?ydrogen', c_lower): normalized.append('H2')
        elif re.search(r'meth.?ane', c_lower): normalized.append('CH4')
        elif re.search(r'acetyl', c_lower): normalized.append('C2H2')
        elif re.search(r'ethyl.?ene', c_lower): normalized.append('C2H4')
        elif re.search(r'ethane', c_lower): normalized.append('C2H6')
        elif re.search(r'carbon.?monoxide|co(?!2)', c_lower): normalized.append('CO')
        elif re.search(r'carbon.?dioxide|co2', c_lower): normalized.append('CO2')
        elif re.search(r'date|time', c_lower): normalized.append('Timestamp')
        else:
            normalized.append(c.strip())
    return normalized


def average_duplicate_columns(df):
    seen, new_cols = {}, []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}.{seen[c]}")
    df.columns = new_cols
    averaged = pd.DataFrame()
    processed = set()

    def base_name(name): return re.sub(r'\.\d+$', '', str(name))
    for col in df.columns:
        name = base_name(col)
        if name in processed: continue
        duplicates = [c for c in df.columns if base_name(c) == name]
        processed.add(name)
        temp = df[duplicates].apply(pd.to_numeric, errors="coerce")
        averaged[name] = temp.mean(axis=1, skipna=True)
    averaged = averaged.dropna(axis=1, how="all")
    averaged = averaged.loc[:, ~averaged.columns.duplicated()]
    return averaged

# =====================================================
# 🗃️ Dataset Manager
# =====================================================
st.header("🗃️ Dataset Manager")

if os.path.exists(DATA_PATH):
    df_data = pd.read_csv(DATA_PATH)
    total_rows = len(df_data)
    transformers = df_data["Transformer"].nunique() if "Transformer" in df_data.columns else 0
    faults = df_data["ExpertLabel"].nunique() if "ExpertLabel" in df_data.columns else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Records", total_rows)
    col2.metric("⚙️ Transformers", transformers)
    col3.metric("💥 Fault Classes", faults)

    if "ExpertLabel" in df_data.columns:
        st.markdown("### Fault Type Distribution")
        fault_counts = df_data["ExpertLabel"].value_counts()
        st.bar_chart(fault_counts)

    with st.expander("⚠️ Advanced Dataset Tools"):
        colA, colB = st.columns(2)
        with colA:
            if st.button("🧹 Delete All Data (Reset)"):
                df_init = pd.DataFrame(columns=df_data.columns)
                df_init.to_csv(DATA_PATH, index=False)
                st.success("✅ Dataset reset successfully.")
        with colB:
            st.download_button("⬇️ Export Current Dataset", data=df_data.to_csv(index=False),
                               file_name="training_data_export.csv", mime="text/csv")

else:
    st.info("No dataset found yet. Upload CSVs below to create one.")

# =====================================================
# 📤 Bulk Upload Section
# =====================================================
st.header("📤 Upload New Datasets")

uploaded_files = st.file_uploader("📂 Upload one or more CSV files", type=["csv"], accept_multiple_files=True)
if uploaded_files:
    combined_data = []
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            df.columns = normalize_column_names(df.columns)
            df = average_duplicate_columns(df)
            name_match = re.search(r'Transformer[_\s-]?([A-Z])', f.name)
            df["Transformer"] = name_match.group(1) if name_match else f.name
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            gases = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]
            df = df[[c for c in df.columns if c in gases or c in ["Timestamp", "Transformer"]]].dropna(subset=gases)
            df["ExpertLabel"] = df.apply(
                lambda r: classify_fault(r.get("H2", 0), r.get("CH4", 0), r.get("C2H2", 0),
                                         r.get("C2H4", 0), r.get("C2H6", 0), r.get("CO", 0), r.get("CO2", 0))[0],
                axis=1)
            combined_data.append(df)
            st.success(f"✅ Processed {f.name} — {df.shape[0]} rows")
        except Exception as e:
            st.error(f"❌ Error processing {f.name}: {e}")

    if combined_data:
        df_combined = pd.concat(combined_data, ignore_index=True)
        df_existing = pd.read_csv(DATA_PATH)
        df_combined["RuleBasedFault"] = df_combined["ExpertLabel"]
        merged = pd.concat([df_existing, df_combined], ignore_index=True)
        merged.to_csv(DATA_PATH, index=False)
        st.success(f"✅ {len(uploaded_files)} files merged. Total dataset size: {merged.shape[0]}")
        st.dataframe(merged.head(10))

# =====================================================
# 🚨 Anomaly Detection (Outlier Scan)
# =====================================================
st.header("🚨 Anomaly Detection Scan")

if os.path.exists(DATA_PATH):
    df_scan = pd.read_csv(DATA_PATH)
    if len(df_scan) > 100:
        gases = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]
        st.write("Running Isolation Forest anomaly scan...")
        model_if = IsolationForest(contamination=0.02, random_state=42)
        df_scan["AnomalyScore"] = model_if.fit_predict(df_scan[gases].fillna(0))
        anomalies = df_scan[df_scan["AnomalyScore"] == -1]
        st.warning(f"⚠️ {len(anomalies)} potential anomalies detected.")
        st.dataframe(anomalies.head(10))
        st.download_button("⬇️ Export Anomaly Report", data=anomalies.to_csv(index=False),
                           file_name="anomaly_report.csv", mime="text/csv")

# =====================================================
# 🤖 Train Model
# =====================================================
st.header("🤖 Train AI Model")

if st.button("Train Model"):
    df = pd.read_csv(DATA_PATH)
    if len(df) < 100:
        st.warning("⚠️ Not enough data to train.")
    elif "ExpertLabel" not in df.columns:
        st.error("❌ Missing ExpertLabel column.")
    else:
        gases = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]
        X, y = df[gases], df["ExpertLabel"]
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")
        st.code(classification_report(y_test, y_pred))
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoder, ENCODER_PATH)

# =====================================================
# 📈 Timeline + Forecast
# =====================================================
st.header("📈 Timeline Intelligence + Forecasting")

if os.path.exists(DATA_PATH):
    df_all = pd.read_csv(DATA_PATH)
    if "Timestamp" in df_all.columns:
        df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"], errors="coerce")

    transformers = sorted(df_all["Transformer"].dropna().unique())
    transformer_choice = st.selectbox("Select Transformer", transformers)
    df_t = df_all[df_all["Transformer"] == transformer_choice].sort_values("Timestamp")

    if len(df_t) > 3:
        gases = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]
        selected_gases = st.multiselect("Select gases", gases, default=["H2", "CH4", "CO", "CO2"])
        fig_gas = px.line(df_t, x="Timestamp", y=selected_gases, title=f"Gas Evolution — Transformer {transformer_choice}")
        st.plotly_chart(fig_gas, use_container_width=True)

        # Predictive Forecast
        st.markdown("### 🔮 Forecast Next 90 Days")
        df_t["Days"] = (df_t["Timestamp"] - df_t["Timestamp"].min()).dt.days
        forecast_results = {}
        for gas in selected_gases:
            X = df_t[["Days"]]; y = df_t[gas]
            if len(X) > 3:
                model_lr = LinearRegression().fit(X, y)
                future_day = np.array([[df_t["Days"].max() + 90]])
                forecast_results[gas] = model_lr.predict(future_day)[0]
        df_forecast = pd.DataFrame.from_dict(forecast_results, orient="index", columns=["Predicted ppm (in 90 days)"])
        st.dataframe(df_forecast.round(2))

        # Risk flag
        risk = df_forecast[df_forecast["Predicted ppm (in 90 days)"] > 1000]
        if not risk.empty:
            st.error(f"⚠️ High risk: {', '.join(risk.index)} gases trending upward.")
        else:
            st.success("✅ Forecast stable. No major rises expected.")

st.markdown("---")
st.caption("Developed by Code GPT 🧑‍💻 | DGA AI v4.7 | Dataset Manager + Anomaly Detection 🌐")
