import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="LiDAR Risk Analysis",
    page_icon="📡",
    layout="wide"
)

MODEL_PATH = Path(__file__).resolve().parent / "lidar_obstacle_model.pkl"

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.metric-card {
    border: 1px solid #d9dce3;
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["model"], saved["features"]

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model, features = load_model()

st.title("📡 LiDAR Obstacle Detection & Risk Analysis")
st.caption("Random Forest classification • Sensor analytics • Distance-based risk assessment")

st.info(
    "This dashboard analyzes uploaded sensor recordings. "
    "It is not a live ESP32 connection or a safety-certified system. "
    "Model labels were originally generated using a 0.50 m distance threshold."
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Dashboard Settings")

mode = st.sidebar.radio(
    "Input mode",
    ["Upload CSV", "Manual Reading"]
)

st.sidebar.subheader("Risk thresholds")

critical_limit = st.sidebar.number_input(
    "Critical: distance at or below (m)",
    min_value=0.01, max_value=10.0, value=0.25, step=0.05
)

high_limit = st.sidebar.number_input(
    "High: distance at or below (m)",
    min_value=0.02, max_value=20.0, value=0.50, step=0.05
)

medium_limit = st.sidebar.number_input(
    "Medium: distance at or below (m)",
    min_value=0.03, max_value=50.0, value=1.00, step=0.10
)

if not critical_limit < high_limit < medium_limit:
    st.sidebar.error("Set thresholds in ascending order: Critical < High < Medium.")
    st.stop()

# ---------------- INPUT ----------------
uploaded = None
manual_data = None

if mode == "Upload CSV":
    uploaded = st.file_uploader(
        "Upload your real_lidar.csv file",
        type=["csv"]
    )

    if uploaded is None:
        st.info("Upload your LiDAR CSV to display predictions, graphs and risk analysis.")
        st.stop()

    try:
        data = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

else:
    st.subheader("Enter a sensor reading")

    c1, c2, c3 = st.columns(3)
    with c1:
        distance = st.number_input(
            "Distance (m)", min_value=0.001,
            max_value=100.0, value=0.35, step=0.05
        )
    with c2:
        angle = st.number_input(
            "Servo angle (degrees)", min_value=0,
            max_value=180, value=90
        )
    with c3:
        signal = st.number_input(
            "Signal strength", min_value=0, value=2000
        )

    manual_data = pd.DataFrame([{
        "distance_m": distance,
        "servo_angle": angle,
        "signal": signal
    }])
    data = manual_data.copy()

# ---------------- DATA VALIDATION ----------------
missing = [c for c in features if c not in data.columns]

if missing:
    st.error(f"Missing required CSV columns: {missing}")
    st.write("Required columns:", features)
    st.stop()

data = data.copy()
data["_original_row"] = np.arange(len(data))

for col in features:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(subset=features)

if "valid" in data.columns:
    data["valid"] = pd.to_numeric(data["valid"], errors="coerce")
    data = data[data["valid"] == 1]

data = data[data["distance_m"] > 0].copy()

if data.empty:
    st.warning("No valid positive-distance readings are available.")
    st.stop()

# ---------------- PREDICTIONS ----------------
data["prediction"] = model.predict(data[features])

probabilities = model.predict_proba(data[features])
data["model_confidence_pct"] = (
    probabilities.max(axis=1) * 100
).round(2)

# Independent, configurable distance-based risk rules
data["risk_level"] = np.select(
    [
        data["distance_m"] <= critical_limit,
        data["distance_m"] <= high_limit,
        data["distance_m"] <= medium_limit
    ],
    ["Critical", "High", "Medium"],
    default="Low"
)

data["recommended_action"] = data["risk_level"].map({
    "Critical": "Stop and inspect immediately",
    "High": "Slow down and check the path",
    "Medium": "Proceed cautiously and monitor",
    "Low": "Continue monitoring"
})

data["obstacle_flag"] = data["prediction"].eq("Obstacle")

# ---------------- TOP METRICS ----------------
st.divider()
st.header("📊 Dashboard Overview")

total = len(data)
obstacle_count = int(data["obstacle_flag"].sum())
clear_count = total - obstacle_count
obstacle_pct = obstacle_count / total * 100
nearest = data["distance_m"].min()
critical_count = int((data["risk_level"] == "Critical").sum())
high_count = int((data["risk_level"] == "High").sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Valid readings", f"{total:,}")
m2.metric("Predicted obstacles", f"{obstacle_count:,}")
m3.metric("Predicted clear", f"{clear_count:,}")
m4.metric("Nearest reading", f"{nearest:.2f} m")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Obstacle prediction rate", f"{obstacle_pct:.1f}%")
m6.metric("Critical risk readings", f"{critical_count:,}")
m7.metric("High risk readings", f"{high_count:,}")
m8.metric("Average distance", f"{data['distance_m'].mean():.2f} m")

# ---------------- CURRENT RISK ----------------
st.header("🚦 Risk Analysis")

if critical_count > 0:
    st.error(
        f"Critical distance band: {critical_count:,} readings at or below "
        f"{critical_limit:.2f} m. Inspect these readings first."
    )
elif high_count > 0:
    st.warning(
        f"High distance band: {high_count:,} readings at or below "
        f"{high_limit:.2f} m. Review these readings carefully."
    )
else:
    st.success(
        "No readings fall within the configured Critical or High distance bands."
    )

r1, r2, r3, r4 = st.columns(4)
r1.metric("Critical", int((data["risk_level"] == "Critical").sum()))
r2.metric("High", int((data["risk_level"] == "High").sum()))
r3.metric("Medium", int((data["risk_level"] == "Medium").sum()))
r4.metric("Low", int((data["risk_level"] == "Low").sum()))

st.caption(
    f"Risk bands: Critical ≤ {critical_limit:.2f} m; "
    f"High ≤ {high_limit:.2f} m; "
    f"Medium ≤ {medium_limit:.2f} m; "
    f"Low > {medium_limit:.2f} m. "
    "These are configurable demonstration thresholds, not validated safety limits."
)

# ---------------- GRAPHS ----------------
st.header("📈 Interactive Graphs")

tab1, tab2, tab3, tab4 = st.tabs([
    "Distance Trend",
    "Prediction Distribution",
    "Risk Distribution",
    "Servo Angle Analysis"
])

with tab1:
    plot_data = data.copy()

    if "time_s" in plot_data.columns:
        plot_data["time_s"] = pd.to_numeric(
            plot_data["time_s"], errors="coerce"
        )
        plot_data = plot_data.dropna(subset=["time_s"])
        x_col = "time_s"
        x_title = "Time (seconds)"
    elif "timestamp" in plot_data.columns:
        x_col = "timestamp"
        x_title = "Timestamp"
    else:
        x_col = "_original_row"
        x_title = "Original row number"

    if len(plot_data) > 5000:
        plot_data = plot_data.iloc[
            np.linspace(0, len(plot_data) - 1, 5000).astype(int)
        ]

    fig = px.line(
        plot_data,
        x=x_col,
        y="distance_m",
        color="risk_level",
        title="LiDAR Distance Over Time",
        labels={
            "distance_m": "Distance (m)",
            x_col: x_title,
            "risk_level": "Risk level"
        },
        hover_data=["servo_angle", "prediction"]
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    pred_counts = (
        data["prediction"].value_counts()
        .rename_axis("Prediction")
        .reset_index(name="Readings")
    )
    fig = px.pie(
        pred_counts,
        names="Prediction",
        values="Readings",
        hole=0.45,
        title="Model Prediction Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(
        data,
        x="distance_m",
        color="prediction",
        nbins=40,
        title="Distance Distribution by Model Prediction",
        labels={"distance_m": "Distance (m)"}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    risk_order = ["Critical", "High", "Medium", "Low"]
    risk_counts = (
        data["risk_level"].value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("Risk level")
        .reset_index(name="Readings")
    )

    fig = px.bar(
        risk_counts,
        x="Risk level",
        y="Readings",
        title="Readings by Distance-Based Risk Level",
        text="Readings",
        category_orders={"Risk level": risk_order}
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(
        data,
        x="risk_level",
        y="distance_m",
        category_orders={"risk_level": risk_order},
        title="Distance Spread by Risk Level"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    angle_data = data.copy()
    angle_data["servo_angle"] = pd.to_numeric(
        angle_data["servo_angle"], errors="coerce"
    )
    angle_data = angle_data.dropna(subset=["servo_angle"])

    angle_summary = angle_data.groupby(
        "servo_angle", as_index=False
    ).agg(
        readings=("distance_m", "size"),
        average_distance=("distance_m", "mean"),
        obstacle_predictions=("obstacle_flag", "sum")
    )

    if not angle_summary.empty:
        fig = px.line(
            angle_summary,
            x="servo_angle",
            y="average_distance",
            markers=True,
            title="Average Distance by Servo Angle",
            labels={
                "servo_angle": "Servo angle (degrees)",
                "average_distance": "Average distance (m)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            angle_summary,
            x="servo_angle",
            y="obstacle_predictions",
            title="Predicted Obstacles by Servo Angle",
            labels={
                "servo_angle": "Servo angle (degrees)",
                "obstacle_predictions": "Obstacle predictions"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------- FILTERS AND TABLE ----------------
st.header("🔎 Reading Explorer")

f1, f2 = st.columns(2)
with f1:
    selected_risks = st.multiselect(
        "Filter risk levels",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium", "Low"]
    )

with f2:
    selected_predictions = st.multiselect(
        "Filter model predictions",
        ["Obstacle", "Clear"],
        default=["Obstacle", "Clear"]
    )

filtered = data[
    data["risk_level"].isin(selected_risks)
    & data["prediction"].isin(selected_predictions)
].copy()

st.write(f"Showing {len(filtered):,} of {total:,} valid readings.")

preferred = [
    "timestamp", "run_id", "time_s", "distance_m",
    "distance_cm", "servo_angle", "signal",
    "prediction", "model_confidence_pct",
    "risk_level", "recommended_action"
]
display_cols = [c for c in preferred if c in filtered.columns]

st.dataframe(
    filtered[display_cols],
    use_container_width=True,
    height=350
)

st.download_button(
    "⬇️ Download Filtered Risk Report",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="lidar_risk_report.csv",
    mime="text/csv"
)

# ---------------- NOTES ----------------
with st.expander("ℹ️ How to interpret this dashboard"):
    st.markdown("""
    - **Prediction:** the Random Forest output trained on distance-derived labels.
    - **Model confidence:** the highest predicted class probability; it is not a safety probability.
    - **Risk level:** a separate rule based on the measured distance and configurable thresholds.
    - **Nearest reading:** the smallest positive distance in the valid dataset.
    - **No live connection:** uploaded CSV readings are historical data, not real-time ESP32 measurements.
    - **Evaluation limitation:** consecutive readings and labels generated from distance can make accuracy appear very high. Test with independently labeled data from new recording sessions before relying on the system.
    """)

st.caption("LiDAR Obstacle Detection • Random Forest • Interactive Risk Analytics")
