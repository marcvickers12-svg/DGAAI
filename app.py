import streamlit as st
import pandas as pd
from datetime import datetime
from dga_logic import classify_fault
import os
import tempfile

# --- Writable data directory (cross-platform safe) ---
TMP_DIR = tempfile.gettempdir()
DATA_DIR = os.path.join(TMP_DIR, "dga_data")
DATA_PATH = os.path.join(DATA_DIR, "training_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)

# Initialize CSV if missing
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
st.title("🧠 DGA AI Training Camp")
st.write("Train your AI with expert fault classification, visualize trends, and filter past entries.")

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

# --- Run Analysis Button ---
if st.button("Run Analysis"):
    fault, explanation = classify_fault(h2, ch4, c2h2, c2h4, c2h6, co, co2)

    st.subheader(f"🧾 Fault Type: {fault}")
    st.caption(explanation)

    expert_label = st.text_input(
        "Expert Label (optional)", placeholder="Enter confirmed fault if different"
    )

    if st.button("💾 Save to Training Data"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            df = pd.read_csv(DATA_PATH)
        except Exception:
            df = pd.DataFrame(columns=[
                "Timestamp", "H2", "CH4", "C2H2", "C2H4", "C2H6",
                "CO", "CO2", "RuleBasedFault", "ExpertLabel"
            ])

        new_row = {
            "Timestamp": timestamp,
            "H2": h2, "CH4": ch4, "C2H2": c2h2, "C2H4": c2h4,
            "C2H6": c2h6, "CO": co, "CO2": co2,
            "RuleBasedFault": fault,
            "ExpertLabel": expert_label if expert_label else fault,
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False)
        st.success(f"✅ Entry saved for AI training! ({timestamp})")

# --- Dataset Preview ---
st.divider()
st.markdown("### 📊 Training Dataset Preview")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    st.dataframe(df.tail(10))

# --- Search and Filter Panel ---
st.divider()
st.markdown("### 🔍 Search & Filter Saved Data")

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)

    # Search box
    search_term = st.text_input("Search by fault type, expert label, or timestamp", placeholder="e.g. T2, D1, PD, 2025-10-17")

    # Range filters
    st.write("Filter by gas range (optional):")
    col1, col2, col3 = st.columns(3)
    with col1:
        h2_min = st.number_input("Min H₂", min_value=0.0, value=0.0)
        h2_max = st.number_input("Max H₂", min_value=0.0, value=10000.0)
    with col2:
        ch4_min = st.number_input("Min CH₄", min_value=0.0, value=0.0)
        ch4_max = st.number_input("Max CH₄", min_value=0.0, value=10000.0)
    with col3:
        fault_filter = st.selectbox(
            "Filter by Rule-Based Fault", ["All"] + sorted(df["RuleBasedFault"].dropna().unique().tolist())
        )

    # Apply filters
    filtered_df = df.copy()

    # Search text filter
    if search_term:
        filtered_df = filtered_df[
            filtered_df.apply(lambda row: row.astype(str).str.contains(search_term, case=False).any(), axis=1)
        ]

    # Numeric filters
    filtered_df = filtered_df[
        (filtered_df["H2"].between(h2_min, h2_max)) &
        (filtered_df["CH4"].between(ch4_min, ch4_max))
    ]

    # Fault type filter
    if fault_filter != "All":
        filtered_df = filtered_df[filtered_df["RuleBasedFault"] == fault_filter]

    st.markdown("#### Filtered Results")
    st.dataframe(filtered_df)

    # Download filtered CSV
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_training_data.csv",
        mime="text/csv"
    )

else:
    st.warning("⚠️ No training data found yet. Add entries first.")

st.markdown("---")
st.caption("Developed by Code GPT 🧑‍💻 | Step 1 of DGA Analysis AI System | v2.1")
