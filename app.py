import streamlit as st
import pandas as pd
from dga_logic import classify_fault
import os

# --- Safe cross-platform data directory setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")

# Ensure directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR, exist_ok=True)

# Ensure CSV file exists
if not os.path.exists(DATA_PATH):
    df_init = pd.DataFrame(
        columns=[
            "H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2",
            "RuleBasedFault", "ExpertLabel"
        ]
    )
    df_init.to_csv(DATA_PATH, index=False)

# --- Streamlit page setup ---
st.set_page_config(page_title="DGA AI Training Camp", layout="centered")
st.title("🧠 DGA AI Training Camp")
st.write("Train your AI with expert fault classification.")

# --- Input fields ---
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

# --- Run Analysis ---
if st.button("Run Analysis"):
    fault, explanation = classify_fault(h2, ch4, c2h2, c2h4, c2h6, co, co2)

    st.subheader(f"🧾 Fault Type: {fault}")
    st.caption(explanation)

    expert_label = st.text_input(
        "Expert Label (optional)", placeholder="Enter confirmed fault if different"
    )

    if st.button("💾 Save to Training Data"):
        try:
            df = pd.read_csv(DATA_PATH)
        except FileNotFoundError:
            df = pd.DataFrame(columns=[
                "H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2", "RuleBasedFault", "ExpertLabel"
            ])

        new_row = {
            "H2": h2,
            "CH4": ch4,
            "C2H2": c2h2,
            "C2H4": c2h4,
            "C2H6": c2h6,
            "CO": co,
            "CO2": co2,
            "RuleBasedFault": fault,
            "ExpertLabel": expert_label if expert_label else fault,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        st.success("✅ Entry saved for AI training!")

# --- Dataset Preview ---
st.divider()
st.markdown("### 📊 Training Dataset Preview")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.dataframe(df.tail(10))

st.markdown("---")
st.caption("Developed by M Vickers 🧑‍💻 | Step 1 of DGA Analysis AI System")
