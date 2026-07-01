import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 1. Load
df = pd.read_excel('../Data/Raw/Data for HIV Study.xlsx', header=[0, 1])

# 2. Define groups
group_columns = {
    "Parts That Tend to Be Most Pain (0: No | 1: Yes | NA: Not Disclosed)",
    "ART?", "Medication Class", "Smoke"
}
valid_sublabels = {
    "Upper Body Pain: Hands/Shoulders/Arms/Neck", "Thoracic Pain: Chest/Upper Back",
    "Lower Body Pain: Low Back/Hip Pain", "Thigh/Knee/Leg Pain", "Feet/Ankle",
    "Drug", "Nucleoside reverse transcriptase inhibitors (NRTIs)",
    "Integrase Strand Transfer Inhibitor (INSTI)",
    "Non-nucleoside reverse transcriptase inhibitor (NNRTI)", "Protease Inhibitor (PI)",
    "Cardiovascular", "Diabetes & Metabolic Conditions", "Pain Relief",
    "Psychiatric Conditions", "Gastrointestinal Conditions", "Infections & Immune Disorders",
    "Respiratory Conditions", "Kidney & Liver Conditions", "Hormonal & Endocrine Conditions",
    "Allergic Conditions", "Others (Vitamins)", "Cigarettes", "E-Cigarettes"
}

# 3. Rename columns
new_col = []
for level0, level1 in df.columns:
    level0_clean = level0.strip()
    level1_clean = str(level1).strip()
    if level0_clean in group_columns and level1_clean in valid_sublabels:
        new_name = f"{level0_clean}_{level1_clean}"
    else:
        new_name = level0_clean
    new_col.append(new_name)

df.columns = new_col

# 4. Drop duplicate columns
df = df.loc[:, ~df.columns.duplicated(keep="first")]

# 4b. Fix ART drug class NA values (must be AFTER renaming)
art_class_columns = [
    "ART?_Nucleoside reverse transcriptase inhibitors (NRTIs)",
    "ART?_Integrase Strand Transfer Inhibitor (INSTI)",
    "ART?_Non-nucleoside reverse transcriptase inhibitor (NNRTI)",
    "ART?_Protease Inhibitor (PI)"
]
for col in art_class_columns:
    df[col] = df[col].replace("NA", 0)
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 5. Fix binary columns (also after renaming)
binary_columns = [
    "Seizure Now", "Neurological Disorder Now", "Neurological Disorder Past",
    "Biological Sex", "Hospitalized (Past Year)", "Injuries/Surgeries"
]
for col in binary_columns:
    df[col] = df[col].replace('n', 0)
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 6. Fix Age
df["Age"] = df["Age"].replace('?', np.nan)
df["Age"] = df["Age"].replace(0, np.nan)
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')

# 7. Separate text columns from numeric columns
# These stay as text - we won't impute them
text_columns = ["ID", "Group", "Ethnicity", "ART?_Drug", "Which medication?",
                "ART?_Nucleoside reverse transcriptase inhibitors (NRTIs)",
                "ART?_Integrase Strand Transfer Inhibitor (INSTI)",
                "ART?_Non-nucleoside reverse transcriptase inhibitor (NNRTI)",
                "ART?_Protease Inhibitor (PI)", "Medications"]

numeric_columns = [col for col in df.columns if col not in text_columns]

# 8. Drop rows where ALL numeric values are missing (completely empty rows)
before = len(df)
df = df.dropna(subset=numeric_columns, how='all')
after = len(df)
print(f"Rows dropped (completely empty): {before - after}")

# 9. Drop rows where more than 50% of numeric values are missing
threshold = len(numeric_columns) * 0.5
before = len(df)
df = df[df[numeric_columns].isnull().sum(axis=1) <= threshold]
after = len(df)
print(f"Rows dropped (over 50% missing): {before - after}")

# 10. KNN Imputation on numeric columns only
imputer = KNNImputer(n_neighbors=5)
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# 11. Round binary columns back to 0 or 1 (KNN gives decimals)
all_binary = [col for col in numeric_columns if col not in
              ["Age", "Weight", "Bodily Pain Rating in Past Week", "Doses ART missed?"]]
for col in all_binary:
    df[col] = df[col].round().astype(int)

# 12. Save cleaned data
df.to_csv('../Data/Cleaned/cleaned_data.csv', index=False)
print(f"Final shape: {df.shape}")
print("Cleaned data saved to Data/Cleaned/cleaned_data.csv")