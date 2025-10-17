
import pandas as pd
import glob
import os
from dga_logic import classify_fault
import re

# === Enhanced UK DGA Dataset Preprocessor ===
# Automatically detects alternate column names and normalizes them to match the expected schema.
# Columns handled: Hydrogen, Methane, Acetylene, Ethylene, Ethane, CO, CO2

def normalize_column_names(columns):
    normalized = []
    for c in columns:
        c_lower = c.lower().strip()
        if re.search(r'h.?ydrogen', c_lower): normalized.append('H2')
        elif re.search(r'meth.?ane', c_lower): normalized.append('CH4')
        elif re.search(r'acetyl', c_lower): normalized.append('C2H2')
        elif re.search(r'ethyl.?ene', c_lower): normalized.append('C2H4')
        elif re.search(r'ethane', c_lower): normalized.append('C2H6')
        elif re.search(r'co2', c_lower): normalized.append('CO2')
        elif re.match(r'co(?!2)', c_lower): normalized.append('CO')
        else: normalized.append(c.strip())
    return normalized


def prepare_uk_dga_dataset(input_folder="./UK_DGA_Files", output_file="UK_DGA_Formatted.csv"):
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not all_files:
        print("❌ No CSV files found in", input_folder)
        return

    df_list = []

    for file in all_files:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"⚠️ Could not read {file}: {e}")
            continue

        # Normalize headers
        df.columns = normalize_column_names(df.columns)

        # Filter only relevant columns
        gases = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]
        df = df[[col for col in df.columns if col in gases]]
        df = df.dropna()

        # Apply classification
        faults = []
        for _, row in df.iterrows():
            try:
                fault, _ = classify_fault(
                    row.get("H2", 0),
                    row.get("CH4", 0),
                    row.get("C2H2", 0),
                    row.get("C2H4", 0),
                    row.get("C2H6", 0),
                    row.get("CO", 0),
                    row.get("CO2", 0)
                )
            except Exception:
                fault = "Unknown"
            faults.append(fault)

        df["ExpertLabel"] = faults
        df_list.append(df)

    # Combine all
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Cleaned & labeled dataset saved: {output_file}")
    print(f"📊 Total rows: {len(merged_df)} | Columns: {list(merged_df.columns)}")


if __name__ == "__main__":
    prepare_uk_dga_dataset()
