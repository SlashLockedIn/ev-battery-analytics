from pathlib import Path
import json
from flask import Flask, render_template
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.io as pio
import joblib
import os
import numpy as np
from flask import request
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
KPI_PATH = PROJECT_DIR / "artifacts" / "dashboard" / "kpi_payload.json"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

def load_kpis():
    if not KPI_PATH.exists():
        raise FileNotFoundError(f"KPI payload not found at: {KPI_PATH}")
    with open(KPI_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

KPI_PATH = PROJECT_DIR / "artifacts" / "dashboard" / "kpi_payload.json"
DATA_PATH = PROJECT_DIR / "data" / "ev_battery_degradation_v1.csv"

# If you already defined app earlier, keep it — just add the functions + route below.

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # minimal type cleanup for plotting
    df["Battery_Status"] = df["Battery_Status"].astype("string").str.strip().str.title()
    df["Driving_Style"] = df["Driving_Style"].astype("string").str.strip().str.title()
    df["Battery_Type"] = df["Battery_Type"].astype("string").str.strip().str.upper()
    df["Car_Model"] = df["Car_Model"].astype("string").str.strip()

    num_cols = [
        "SoH_Percent","Total_Charging_Cycles","Avg_Temperature_C","Fast_Charge_Ratio",
        "Avg_Discharge_Rate_C","Internal_Resistance_Ohm","Vehicle_Age_Months","Battery_Capacity_kWh"
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["SoH_Percent","Battery_Status"])
    return df

MODEL_DIR = PROJECT_DIR / "artifacts" / "models"


ENCODER_PATH = MODEL_DIR / "status_encoder.pkl"
REG_PIPE_PATH = MODEL_DIR / "soh_reg_pipeline.pkl"
CLF_PIPE_PATH = MODEL_DIR / "status_clf_pipeline.pkl"

reg_pipe = joblib.load(REG_PIPE_PATH)
clf_pipe = joblib.load(CLF_PIPE_PATH)


status_encoder = joblib.load(ENCODER_PATH)



@app.route("/")
def home():
    # redirect-ish: just render dashboard directly
    kpis = load_kpis()
    return render_template("dashboard.html", kpis=kpis)

@app.route("/dashboard")
def dashboard():
    kpis = load_kpis()
    return render_template("dashboard.html", kpis=kpis)

@app.route("/analytics")
def analytics():
    df = load_data()

    # ---------- Chart 1: Battery Status Distribution ----------
    status_counts = df["Battery_Status"].value_counts().reset_index()
    status_counts.columns = ["Battery_Status", "Count"]
    fig1 = px.bar(
        status_counts, x="Battery_Status", y="Count",
        title="Battery Status Mix (Fleet Risk Distribution)"
    )

    # ---------- Chart 2: SoH Distribution ----------
    fig2 = px.histogram(
        df, x="SoH_Percent", nbins=25,
        title="State of Health (SoH %) Distribution"
    )

    # ---------- Chart 3: Temperature vs SoH (colored by status) ----------
    fig3 = px.scatter(
        df.sample(min(800, len(df)), random_state=42),
        x="Avg_Temperature_C", y="SoH_Percent",
        color="Battery_Status",
        title="Thermal Stress Impact: Avg Temperature vs SoH",
        hover_data=["Battery_Type","Driving_Style","Total_Charging_Cycles"]
    )

    # ---------- Chart 4: Charging Cycles vs SoH (colored by fast charge band) ----------
    df["FastCharge_Band"] = pd.cut(
        df["Fast_Charge_Ratio"],
        bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
        labels=["Low (<=0.2)", "Medium (0.2-0.5)", "High (0.5-0.8)", "Very High (0.8+)"]
    )
    fig4 = px.scatter(
        df.sample(min(800, len(df)), random_state=7),
        x="Total_Charging_Cycles", y="SoH_Percent",
        color="FastCharge_Band",
        title="Aging Curve: Charging Cycles vs SoH (by Fast-Charge Behavior)",
        hover_data=["Avg_Temperature_C","Internal_Resistance_Ohm","Battery_Status"]
    )

    # ---------- Chart 5: Internal Resistance by Battery Status (box plot) ----------
    fig5 = px.box(
        df,
        x="Battery_Status", y="Internal_Resistance_Ohm",
        title="Internal Resistance vs Failure Risk (Battery Status)"
    )

    # ---------- Chart 6: Segment view – Avg SoH by Driving Style & Battery Type ----------
    seg = (df.groupby(["Driving_Style","Battery_Type"], dropna=True)
             .agg(avg_soh=("SoH_Percent","mean"),
                  n=("SoH_Percent","size"),
                  critical_share=("Battery_Status", lambda s: (s=="Critical").mean()*100))
             .reset_index())
    fig6 = px.scatter(
        seg, x="avg_soh", y="critical_share",
        size="n", color="Driving_Style",
        facet_col="Battery_Type",
        title="Segment Risk Map: Avg SoH vs Critical Share (bubble size = volume)",
        labels={"avg_soh":"Avg SoH (%)", "critical_share":"Critical Share (%)"}
    )

    # Convert figures to HTML snippets (no full html wrapper)
    charts = {
        "fig1": pio.to_html(fig1, full_html=False, include_plotlyjs="cdn"),
        "fig2": pio.to_html(fig2, full_html=False, include_plotlyjs=False),
        "fig3": pio.to_html(fig3, full_html=False, include_plotlyjs=False),
        "fig4": pio.to_html(fig4, full_html=False, include_plotlyjs=False),
        "fig5": pio.to_html(fig5, full_html=False, include_plotlyjs=False),
        "fig6": pio.to_html(fig6, full_html=False, include_plotlyjs=False),
    }

    return render_template("analytics.html", charts=charts)

from flask import request

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        # Build a 1-row dataframe with raw feature names used in training
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

        # Debug prints (optional)
        print("REG MODEL TYPE:", type(reg_pipe))
        print("INPUT ROW SHAPE:", row.shape)
        print("INPUT ROW COLS:", list(row.columns))
        print("FORM KEYS:", list(request.form.keys()))

        soh_pred = float(reg_pipe.predict(row)[0])
        status_label = str(clf_pipe.predict(row)[0])  # pipeline returns label string directly

        prediction = {
            "soh": round(soh_pred, 2),
            "status": status_label
        }

    return render_template("predictions.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)