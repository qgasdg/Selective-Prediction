"""
Data loading, cleaning, and feature engineering for the Diabetes 130-US dataset.

Pipeline summary:
  1. Remove dead / hospice patients (discharge_disposition_id in DEAD_IDS)
  2. Keep only each patient's first encounter (dedup on patient_nbr)
  3. Binarize target: readmitted == '<30' → 1, else 0
  4. Drop weight (96.9% missing), encounter_id, patient_nbr
  5. Ordinal-encode age, drug columns, lab results
  6. Group ICD-9 diagnosis codes into 9 broad categories
  7. One-hot encode remaining categorical columns
  8. Stratified 70 / 15 / 15 train / val / test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Constants ────────────────────────────────────────────────────────────────

DEAD_IDS = {11, 13, 14, 19, 20, 21}  # death or hospice discharge

AGE_MAP = {
    '[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4,
    '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9,
}

DRUG_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
    'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone',
]
DRUG_ORDINAL = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': 3}

A1C_MAP = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
GLU_MAP  = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}

# ── ICD-9 grouping ───────────────────────────────────────────────────────────

def _icd9_group(code: str) -> str:
    """Map a raw ICD-9 string to one of 9 broad disease categories."""
    if pd.isna(code) or str(code).strip() in ('?', ''):
        return 'Missing'
    code = str(code).strip()
    if code.startswith('V'):
        return 'Supplementary'
    if code.startswith('E'):
        return 'External'
    try:
        c = float(code)
    except ValueError:
        return 'Other'

    if 250 <= c < 251:           return 'Diabetes'
    if 390 <= c <= 459 or c == 785: return 'Circulatory'
    if 460 <= c <= 519 or c == 786: return 'Respiratory'
    if 520 <= c <= 579 or c == 787: return 'Digestive'
    if 580 <= c <= 629 or c == 788: return 'Genitourinary'
    if 710 <= c <= 739:          return 'Musculoskeletal'
    if 140 <= c <= 239:          return 'Neoplasm'
    if 800 <= c <= 999:          return 'Injury'
    return 'Other'

# ── Step 1-3: load & clean ───────────────────────────────────────────────────

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load raw CSV and apply the three structural cleaning steps described in
    report section 2.3.
    """
    df = pd.read_csv(filepath, na_values='?', low_memory=False)
    original_rows = len(df)

    # (1) Remove dead / hospice discharges
    df = df[~df['discharge_disposition_id'].isin(DEAD_IDS)].copy()
    print(f"After removing dead/hospice: {len(df):,} rows "
          f"(removed {original_rows - len(df):,})")

    # (2) Keep each patient's first encounter only
    df = (df.sort_values('encounter_id')
            .groupby('patient_nbr', as_index=False)
            .first())
    print(f"After dedup (first encounter per patient): {len(df):,} rows")

    # (3) Binarize target
    df['target'] = (df['readmitted'] == '<30').astype(int)
    pos = df['target'].sum()
    print(f"Target distribution  →  positive: {pos:,} ({pos/len(df):.1%})  "
          f"negative: {len(df)-pos:,} ({1-pos/len(df):.1%})")

    return df

# ── Step 4-7: feature engineering ───────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Drop columns ────────────────────────────────────────────────────────
    always_drop = ['encounter_id', 'patient_nbr', 'readmitted', 'weight']
    # examide and citoglipton are always 'No' in this dataset → zero variance
    zero_var = [c for c in ('examide', 'citoglipton') if c in df.columns]
    df = df.drop(columns=[c for c in always_drop + zero_var if c in df.columns])

    # ── Gender ──────────────────────────────────────────────────────────────
    df = df[df['gender'].isin(['Male', 'Female'])].copy()
    df['gender'] = (df['gender'] == 'Male').astype(int)

    # ── Age (ordinal) ────────────────────────────────────────────────────────
    df['age'] = df['age'].map(AGE_MAP)

    # ── Race (fill missing) ──────────────────────────────────────────────────
    df['race'] = df['race'].fillna('Unknown')

    # ── High-missing categoricals ─────────────────────────────────────────
    df['medical_specialty'] = df['medical_specialty'].fillna('Unknown')
    df['payer_code']        = df['payer_code'].fillna('Unknown')

    # ── ICD-9 diagnosis grouping ─────────────────────────────────────────
    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].apply(_icd9_group)

    # ── Drug columns (ordinal) ───────────────────────────────────────────
    present_drugs = [c for c in DRUG_COLS if c in df.columns and c not in zero_var]
    for col in present_drugs:
        df[col] = df[col].map(DRUG_ORDINAL).fillna(0).astype(int)

    # ── Lab results (ordinal) ────────────────────────────────────────────
    df['A1Cresult']     = df['A1Cresult'].map(A1C_MAP).fillna(0).astype(int)
    df['max_glu_serum'] = df['max_glu_serum'].map(GLU_MAP).fillna(0).astype(int)

    # ── Binary flags ────────────────────────────────────────────────────
    df['change']      = (df['change'] == 'Ch').astype(int)
    df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)

    # ── One-hot encode remaining object columns ───────────────────────────
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'target']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)

    # ── Final NaN fill ────────────────────────────────────────────────────
    df = df.fillna(0)

    print(f"Feature matrix shape: {df.drop(columns=['target']).shape}")
    return df

# ── Step 8: split ────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame,
               val_size: float = 0.15,
               test_size: float = 0.15,
               random_state: int = 42):
    """Stratified 70/15/15 train/val/test split."""
    X = df.drop(columns=['target'])
    y = df['target']

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=random_state
    )

    for name, split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        print(f"{name}: {len(split):,} rows  "
              f"(pos {split.mean():.1%})")

    return X_train, X_val, X_test, y_train, y_val, y_test
