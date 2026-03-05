from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

NUM_COLS = [
    "Battery_Capacity_kWh","Vehicle_Age_Months","Total_Charging_Cycles",
    "Avg_Temperature_C","Fast_Charge_Ratio","Avg_Discharge_Rate_C","Internal_Resistance_Ohm"
]
CAT_COLS = ["Battery_Type","Driving_Style","Car_Model"]  # keep if you trained with it

FEATURES = NUM_COLS + CAT_COLS

def train_if_missing(project_dir: Path) -> None:
    model_dir = project_dir / "artifacts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    reg_path = model_dir / "soh_reg_pipeline.pkl"
    clf_path = model_dir / "status_clf_pipeline.pkl"

    # If already present, do nothing
    if reg_path.exists() and clf_path.exists():
        return

    data_path = project_dir / "data" / "raw" / "ev_battery_degradation_v1.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset missing at {data_path}. Needed to train models on startup.")

    df = pd.read_csv(data_path).drop_duplicates()

    # basic cleaning
    for c in ["Battery_Type","Driving_Style","Car_Model","Battery_Status"]:
        df[c] = df[c].astype("string").str.strip()

    for c in NUM_COLS + ["SoH_Percent"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=FEATURES + ["SoH_Percent","Battery_Status"])

    X = df[FEATURES]
    y_reg = df["SoH_Percent"]
    y_clf = df["Battery_Status"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ]
    )

    reg_pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1))
    ])
    reg_pipe.fit(X, y_reg)

    clf_pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1))
    ])
    clf_pipe.fit(X, y_clf)

    joblib.dump(reg_pipe, reg_path, compress=3)
    joblib.dump(clf_pipe, clf_path, compress=3)