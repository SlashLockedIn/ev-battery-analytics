from pathlib import Path
import os
import json

import pandas as pd
import numpy as np

from flask import Flask, render_template, request

import plotly.express as px
import plotly.io as pio

import joblib

# -----------------------------
# Paths (Render-safe)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../webapp
PROJECT_DIR = BASE_DIR.parent                      # project root
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

KPI_PATH = PROJECT_DIR / "artifacts" / "dashboard" / "kpi_payload.json"
DATA_PATH = PROJECT_DIR / "data" / "ev_battery_degradation_v1.csv"
MODEL_DIR = PROJECT_DIR / "artifacts" / "models"

REG_MODEL_PATH = MODEL_DIR / "soh_reg_pipeline.pkl"
CLF_MODEL_PATH = MODEL_DIR / "status_clf_pipeline.pkl"

# -----------------------------
# Flask app
# -----------------------------
app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)

# -----------------------------
# Helpers: KPI + Data
# -----------------------------
def load_kpis() -> dict:
    if not KPI_PATH.exists():
        raise FileNotFoundError(f"KPI payload not found at: {KPI_PATH}")
    with open(KPI_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH).drop_duplicates()

    # Minimal cleanup for consistent plotting
    for col in ["Battery_Status", "Driving_Style", "Battery_Type", "Car_Model"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()

    # normalize casing
    if "Battery_Status" in df.columns:
        df["Battery_Status"] = df["Battery_Status"].str.title()
    if "Driving_Style" in df.columns:
        df["Driving_Style"] = df["Driving_Style"].str.title()
    if "Battery_Type" in df.columns:
        df["Battery_Type"] = df["Battery_Type"].str.upper()

    num_cols = [
        "SoH_Percent",
        "Total_Charging_Cycles",
        "Avg_Temperature_C",
        "Fast_Charge_Ratio",
        "Avg_Discharge_Rate_C",
        "Internal_Resistance_Ohm",
        "Vehicle_Age_Months",
        "Battery_Capacity_kWh",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only rows needed for charts
    if "SoH_Percent" in df.columns and "Battery_Status" in df.columns:
        df = df.dropna(subset=["SoH_Percent", "Battery_Status"])

    return df

# -----------------------------
# Lazy model loader (NO import-time loads)
# -----------------------------
_reg_pipe = None
_clf_pipe = None

def get_pipelines():
    """
    Ensures models exist (train if missing), then loads them once.
    This function must be called only when you actually need predictions.
    """
    global _reg_pipe, _clf_pipe

    if _reg_pipe is not None and _clf_pipe is not None:
        return _reg_pipe, _clf_pipe

    # Ensure model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # If models missing, train them now (Render-safe)
    if (not REG_MODEL_PATH.exists()) or (not CLF_MODEL_PATH.exists()):
        from train_models import train_if_missing
        train_if_missing(PROJECT_DIR)

    # Load after training (or if already present)
    _reg_pipe = joblib.load(REG_MODEL_PATH)
    _clf_pipe = joblib.load(CLF_MODEL_PATH)

    return _reg_pipe, _clf_pipe

# -----------------------------
# Routes
# -----------------------------
@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/")
def home():
    kpis = load_kpis()
    return render_template("dashboard.html", kpis=kpis)

@app.route("/dashboard")
def dashboard():
    kpis = load_kpis()
    return render_template("dashboard.html", kpis=kpis)

@app.route("/analytics")
def analytics():
    df = load_data()

    # Chart 1: Battery Status Distribution
    status_counts = df["Battery_Status"].value_counts().reset_index()
    status_counts.columns = ["Battery_Status", "Count"]
    fig1 = px.bar(
        status_counts, x="Battery_Status", y="Count",
        title="Battery Status Mix (Fleet Risk Distribution)"
    )

    # Chart 2: SoH Distribution
    fig2 = px.histogram(
        df, x="SoH_Percent", nbins=25,
        title="State of Health (SoH %) Distribution"
    )

    # Chart 3: Temperature vs SoH
    sample_df = df.sample(min(800, len(df)), random_state=42) if len(df) > 0 else df
    fig3 = px.scatter(
        sample_df,
        x="Avg_Temperature_C", y="SoH_Percent",
        color="Battery_Status",
        title="Thermal Stress Impact: Avg Temperature vs SoH",
        hover_data=["Battery_Type", "Driving_Style", "Total_Charging_Cycles"]
    )

    # Chart 4: Charging cycles vs SoH by fast charge band
    df2 = df.copy()
    df2["FastCharge_Band"] = pd.cut(
        df2["Fast_Charge_Ratio"],
        bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
        labels=["Low (<=0.2)", "Medium (0.2-0.5)", "High (0.5-0.8)", "Very High (0.8+)"]
    )
    sample_df2 = df2.sample(min(800, len(df2)), random_state=7) if len(df2) > 0 else df2
    fig4 = px.scatter(
        sample_df2,
        x="Total_Charging_Cycles", y="SoH_Percent",
        color="FastCharge_Band",
        title="Aging Curve: Charging Cycles vs SoH (by Fast-Charge Behavior)",
        hover_data=["Avg_Temperature_C", "Internal_Resistance_Ohm", "Battery_Status"]
    )

    # Chart 5: Internal Resistance by Battery Status
    fig5 = px.box(
        df,
        x="Battery_Status", y="Internal_Resistance_Ohm",
        title="Internal Resistance vs Failure Risk (Battery Status)"
    )

    # Chart 6: Segment risk map
    seg = (df.groupby(["Driving_Style", "Battery_Type"], dropna=True)
             .agg(avg_soh=("SoH_Percent", "mean"),
                  n=("SoH_Percent", "size"),
                  critical_share=("Battery_Status", lambda s: (s == "Critical").mean() * 100))
             .reset_index())

    fig6 = px.scatter(
        seg, x="avg_soh", y="critical_share",
        size="n", color="Driving_Style",
        facet_col="Battery_Type",
        title="Segment Risk Map: Avg SoH vs Critical Share (bubble size = volume)",
        labels={"avg_soh": "Avg SoH (%)", "critical_share": "Critical Share (%)"}
    )

    charts = {
        "fig1": pio.to_html(fig1, full_html=False, include_plotlyjs="cdn"),
        "fig2": pio.to_html(fig2, full_html=False, include_plotlyjs=False),
        "fig3": pio.to_html(fig3, full_html=False, include_plotlyjs=False),
        "fig4": pio.to_html(fig4, full_html=False, include_plotlyjs=False),
        "fig5": pio.to_html(fig5, full_html=False, include_plotlyjs=False),
        "fig6": pio.to_html(fig6, full_html=False, include_plotlyjs=False),
    }

    return render_template("analytics.html", charts=charts)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error_msg = None

    if request.method == "POST":
        try:
            reg_pipe, clf_pipe = get_pipelines()

            row = pd.DataFrame([{
                "Battery_Capacity_kWh": float(request.form["capacity"]),
                "Vehicle_Age_Months": float(request.form["age"]),
                "Total_Charging_Cycles": float(request.form["cycles"]),
                "Avg_Temperature_C": float(request.form["temperature"]),
                "Fast_Charge_Ratio": float(request.form["fastcharge"]),
                "Avg_Discharge_Rate_C": float(request.form["discharge"]),
                "Internal_Resistance_Ohm": float(request.form["resistance"]),
                "Battery_Type": request.form["battery_type"].strip(),
                "Driving_Style": request.form["driving_style"].strip(),
                "Car_Model": request.form["car_model"].strip(),
            }])

            soh_pred = float(reg_pipe.predict(row)[0])
            status_label = str(clf_pipe.predict(row)[0])

            prediction = {
                "soh": round(soh_pred, 2),
                "status": status_label
            }
        except Exception as e:
            # show something on UI instead of blank 502
            error_msg = str(e)

    return render_template("predictions.html", prediction=prediction, error_msg=error_msg)

# Local dev only (Render uses gunicorn)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)