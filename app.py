# =====================================================
# 📤 Bulk Upload + Auto-Heal + Visualization
# =====================================================
st.header("📤 Bulk Upload DGA Datasets (Auto-Heal Mode)")

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
    seen = {}
    new_cols = []
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

    def base_name(name):
        return re.sub(r'\.\d+$', '', str(name))

    for col in df.columns:
        name = base_name(col)
        if name in processed:
            continue

        duplicates = [c for c in df.columns if base_name(c) == name]
        processed.add(name)

        try:
            temp = df[duplicates].apply(pd.to_numeric, errors="coerce")
            averaged[name] = temp.mean(axis=1, skipna=True)
            if averaged[name].isna().all():
                averaged[name] = df[duplicates[0]]
        except Exception as e:
            st.warning(f"⚠️ Could not process {name}: {e}")
            averaged[name] = df[duplicates[0]]

    averaged = averaged.dropna(axis=1, how="all")
    averaged = averaged.loc[:, ~averaged.columns.duplicated()]
    return averaged


uploaded_files = st.file_uploader(
    "📂 Upload one or more CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    combined_data = []
    for uploaded_file in uploaded_files:
        try:
            df_upload = pd.read_csv(uploaded_file)
            df_upload.columns = normalize_column_names(df_upload.columns)
            df_upload = average_duplicate_columns(df_upload)

            # Extract transformer name from filename
            name_match = re.search(r'Transformer[_\s-]?([A-Z])', uploaded_file.name)
            df_upload["Transformer"] = name_match.group(1) if name_match else uploaded_file.name

            if "Timestamp" in df_upload.columns:
                df_upload["Timestamp"] = pd.to_datetime(df_upload["Timestamp"], errors="coerce")

            gases = ["H2", "CH4", "C2H2", "C2H4", "C2H6", "CO", "CO2"]
            df_upload = df_upload[[col for col in df_upload.columns if col in gases or col == "Timestamp" or col == "Transformer"]]
            df_upload = df_upload.dropna(subset=gases)

            # Apply classification
            df_upload["ExpertLabel"] = df_upload.apply(
                lambda row: classify_fault(
                    row.get("H2", 0), row.get("CH4", 0),
                    row.get("C2H2", 0), row.get("C2H4", 0),
                    row.get("C2H6", 0), row.get("CO", 0), row.get("CO2", 0)
                )[0], axis=1
            )
            combined_data.append(df_upload)

            st.success(f"✅ Processed {uploaded_file.name} — {df_upload.shape[0]} rows")
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

    if combined_data:
        df_combined = pd.concat(combined_data, ignore_index=True)
        st.success(f"✅ All files processed! Total rows: {df_combined.shape[0]}")

        st.dataframe(df_combined.head(10))

        # Analytics
        st.markdown("### 🔍 Transformer-wise Breakdown")
        st.write(df_combined.groupby("Transformer").size())

        st.markdown("### 🥧 Fault Distribution Across Transformers")
        fig_fault = px.pie(df_combined, names="ExpertLabel", title="Fault Type Distribution")
        st.plotly_chart(fig_fault, use_container_width=True)

        if st.button("💾 Merge All with Training Data"):
            df_existing = pd.read_csv(DATA_PATH)
            df_combined["RuleBasedFault"] = df_combined["ExpertLabel"]
            merged = pd.concat([df_existing, df_combined], ignore_index=True)
            merged.to_csv(DATA_PATH, index=False)
            st.success(f"✅ {len(uploaded_files)} files merged into training dataset successfully!")
