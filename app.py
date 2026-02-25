"""
⚡ EV Route Energy Predictor — Streamlit Demo
=============================================
Run with:  streamlit run app.py
Requires:  pip install streamlit xgboost plotly pandas numpy joblib
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Energy Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Dark background */
  .stApp {
    background-color: #0d0f14;
    color: #e8eaf0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #1e2330;
  }
  section[data-testid="stSidebar"] * {
    color: #c8ccd8 !important;
  }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #13161e 0%, #1a1f2e 100%);
    border: 1px solid #1e2a3a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4aa, #0099ff);
  }
  .metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5a6280 !important;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #00d4aa !important;
    line-height: 1.1;
  }
  .metric-unit {
    font-size: 13px;
    color: #5a6280 !important;
    margin-top: 4px;
  }
  .metric-negative {
    color: #00bfff !important;
  }

  /* Section headers */
  .section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3a4060;
    padding: 6px 0;
    border-bottom: 1px solid #1a1f30;
    margin-bottom: 16px;
  }

  /* Segment table */
  .seg-row {
    background: #13161e;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  /* Warning/info boxes */
  .info-box {
    background: #0a1628;
    border-left: 3px solid #0099ff;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    color: #8899bb;
    margin: 12px 0;
  }
  .warn-box {
    background: #1a1200;
    border-left: 3px solid #ffaa00;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    color: #bb9933;
    margin: 12px 0;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #00d4aa, #0099ff);
    color: #0d0f14 !important;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
    font-weight: 700;
    padding: 10px 20px;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  /* Slider labels */
  .stSlider label, .stSelectbox label, .stNumberInput label {
    color: #8899bb !important;
    font-size: 13px !important;
  }

  /* Title */
  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 36px;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4aa, #0099ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    line-height: 1.1;
  }
  .hero-sub {
    color: #4a5070;
    font-size: 14px;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
  }

  /* Plotly chart background */
  .js-plotly-plot { border-radius: 12px; overflow: hidden; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
    background: #13161e;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #4a5070;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 1px;
  }
  .stTabs [aria-selected="true"] {
    background: #1e2535 !important;
    color: #00d4aa !important;
  }

  /* Divider */
  hr { border-color: #1e2330 !important; }
</style>
""", unsafe_allow_html=True)


# ── Vehicle & Surface Configs ──────────────────────────────────────────────────
VEHICLE_SPECS = {
    "🚗 Compact":  {"mass": 1600, "Cd": 0.28, "A": 2.2, "regen": 0.70, "encoded": 0},
    "🚙 Sedan":    {"mass": 1900, "Cd": 0.23, "A": 2.4, "regen": 0.75, "encoded": 1},
    "🚕 SUV":      {"mass": 2400, "Cd": 0.33, "A": 2.8, "regen": 0.65, "encoded": 2},
    "🛻 Pickup":   {"mass": 2900, "Cd": 0.40, "A": 3.2, "regen": 0.55, "encoded": 3},
}
SURFACE_SPECS = {
    "Smooth Asphalt":  {"encoded": 0, "Crr": 0.008, "emoji": "🛣️"},
    "Worn Asphalt":    {"encoded": 1, "Crr": 0.011, "emoji": "🚧"},
    "Cobblestone":     {"encoded": 2, "Crr": 0.018, "emoji": "🪨"},
    "Gravel":          {"encoded": 3, "Crr": 0.025, "emoji": "⛰️"},
    "Dirt Road":       {"encoded": 4, "Crr": 0.030, "emoji": "🌱"},
}


# ── Feature Engineering (mirrors training) ────────────────────────────────────
def build_features(distance_km, slope_deg, speed_kmh, bump_density,
                   surface_key, vehicle_key, temperature_c):
    v = VEHICLE_SPECS[vehicle_key]
    s = SURFACE_SPECS[surface_key]
    speed_ms  = speed_kmh / 3.6
    slope_rad = np.radians(slope_deg)
    delta_h   = distance_km * 1000 * np.sin(slope_rad)

    return {
        "vehicle_mass_kg":        v["mass"],
        "drag_coefficient":       v["Cd"],
        "frontal_area_m2":        v["A"],
        "regen_efficiency":       v["regen"],
        "distance_m":             distance_km * 1000,
        "distance_km":            distance_km,
        "slope_deg":              slope_deg,
        "slope_pct":              np.tan(slope_rad) * 100,
        "elevation_change_m":     delta_h,
        "bump_density_per_km":    bump_density,
        "speed_kmh":              speed_kmh,
        "temperature_c":          temperature_c,
        "surface_encoded":        s["encoded"],
        "vehicle_encoded":        v["encoded"],
        "aero_power_proxy":       0.5 * 1.225 * v["Cd"] * v["A"] * speed_ms**3,
        "is_uphill":              int(slope_deg > 1),
        "is_downhill":            int(slope_deg < -1),
        "is_flat":                int(-1 <= slope_deg <= 1),
        "bump_speed_interaction": bump_density * speed_kmh,
        "slope_mass_interaction": slope_deg * v["mass"],
        "regen_potential":        v["regen"] * abs(slope_deg) if slope_deg < -1 else 0,
        "temp_deviation":         abs(temperature_c - 20),
        "slope_abs":              abs(slope_deg),
        "speed_squared":          speed_ms ** 2,
    }


FEATURE_COLS = [
    "vehicle_mass_kg", "drag_coefficient", "frontal_area_m2", "regen_efficiency",
    "distance_m", "distance_km", "slope_deg", "slope_pct", "elevation_change_m",
    "bump_density_per_km", "speed_kmh", "temperature_c", "surface_encoded",
    "vehicle_encoded", "aero_power_proxy", "is_uphill", "is_downhill", "is_flat",
    "bump_speed_interaction", "slope_mass_interaction", "regen_potential",
    "temp_deviation", "slope_abs", "speed_squared",
]


# ── Model Loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ev_xgboost_model.pkl")
        return model, None
    except FileNotFoundError:
        try:
            model = xgb.XGBRegressor()
            model.load_model("ev_xgboost_model.json")
            return model, None
        except FileNotFoundError:
            return None, "Model file not found. Place **ev_xgboost_model.pkl** or **ev_xgboost_model.json** in the same folder as app.py."

model, model_error = load_model()


def predict(distance_km, slope_deg, speed_kmh, bump_density,
            surface_key, vehicle_key, temperature_c):
    if model is None:
        return 0, 0
    feats  = build_features(distance_km, slope_deg, speed_kmh, bump_density,
                            surface_key, vehicle_key, temperature_c)
    X      = pd.DataFrame([feats])[FEATURE_COLS]
    wh_km  = float(model.predict(X)[0])
    total  = wh_km * distance_km
    return wh_km, total


# ── Session State ──────────────────────────────────────────────────────────────
if "segments" not in st.session_state:
    st.session_state.segments = []


# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Segment Builder
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="hero-title">⚡ EV</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ENERGY PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if model_error:
        st.error(model_error)

    st.markdown('<div class="section-header">ADD ROAD SEGMENT</div>', unsafe_allow_html=True)

    vehicle_key = st.selectbox("Vehicle", list(VEHICLE_SPECS.keys()))
    surface_key = st.selectbox("Road Surface", list(SURFACE_SPECS.keys()))

    col1, col2 = st.columns(2)
    with col1:
        distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=50.0,
                                       value=5.0, step=0.5)
    with col2:
        speed_kmh = st.slider("Speed (km/h)", 10, 130, 80)

    slope_deg = st.slider("Slope (°)", -20.0, 20.0, 0.0, step=0.5,
                           help="Negative = downhill, Positive = uphill")

    bump_density = st.slider("Bump Density (bumps/km)", 0, 100, 5)
    temperature_c = st.slider("Temperature (°C)", -15, 45, 22)

    # Live preview
    wh_km_preview, total_preview = predict(
        distance_km, slope_deg, speed_kmh, bump_density,
        surface_key, vehicle_key, temperature_c
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if wh_km_preview < 0:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Segment Preview</div>
          <div class="metric-value metric-negative">{wh_km_preview:.0f}</div>
          <div class="metric-unit">Wh/km · Net regen</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Segment Preview</div>
          <div class="metric-value">{wh_km_preview:.0f}</div>
          <div class="metric-unit">Wh/km · {total_preview:.0f} Wh total</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("＋  ADD SEGMENT TO ROUTE"):
        st.session_state.segments.append({
            "vehicle":      vehicle_key,
            "surface":      surface_key,
            "distance_km":  distance_km,
            "speed_kmh":    speed_kmh,
            "slope_deg":    slope_deg,
            "bump_density": bump_density,
            "temperature_c":temperature_c,
            "wh_km":        wh_km_preview,
            "total_wh":     total_preview,
        })
        st.rerun()

    if st.session_state.segments:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑  CLEAR ROUTE"):
            st.session_state.segments = []
            st.rerun()


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ════════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 32px 0 24px 0;'>
  <div class='hero-title' style='font-size:42px;'>EV Route Energy Predictor</div>
  <div class='hero-sub' style='margin-top:10px; font-size:13px; color:#3a4a70;'>
    PHYSICS-INFORMED · XGBOOST · ROAD TOPOLOGY AWARE
  </div>
</div>
""", unsafe_allow_html=True)

if model_error:
    st.error(f"⚠️ {model_error}")

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.segments:
    st.markdown("""
    <div style='text-align:center; padding: 80px 40px; color: #2a3050;'>
      <div style='font-size: 64px; margin-bottom: 20px;'>🛣️</div>
      <div style='font-family: Space Mono, monospace; font-size: 18px; color: #3a4a70;'>
        Build your route
      </div>
      <div style='font-size: 14px; color: #2a3050; margin-top: 10px;'>
        Add road segments from the sidebar to simulate energy consumption.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Compute trip totals ────────────────────────────────────────────────────────
segs = pd.DataFrame(st.session_state.segments)
total_distance  = segs["distance_km"].sum()
total_energy    = segs["total_wh"].sum()
avg_wh_km       = total_energy / total_distance if total_distance > 0 else 0
total_regen     = segs.loc[segs["total_wh"] < 0, "total_wh"].sum()
net_uphill      = segs.loc[segs["slope_deg"] > 0, "slope_deg"].mean()
km_equivalent   = total_energy / 1000  # rough kWh


# ── KPI Row ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, "TOTAL DISTANCE",  f"{total_distance:.1f}", "km"),
    (c2, "TOTAL ENERGY",    f"{total_energy/1000:.2f}", "kWh"),
    (c3, "AVG CONSUMPTION", f"{avg_wh_km:.0f}", "Wh/km"),
    (c4, "REGEN RECOVERED", f"{abs(total_regen)/1000:.3f}", "kWh"),
    (c5, "SEGMENTS",        f"{len(segs)}", "road sections"),
]
for col, label, val, unit in kpis:
    with col:
        color = "#00bfff" if label == "REGEN RECOVERED" else "#00d4aa"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color};">{val}</div>
          <div class="metric-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  ROUTE ANALYSIS", "🗺️  SEGMENT BREAKDOWN", "⚙️  PHYSICS DETAIL"])

PLOT_LAYOUT = dict(
    paper_bgcolor="#0d0f14",
    plot_bgcolor="#0d0f14",
    font=dict(family="DM Sans", color="#8899bb", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1a1f30", linecolor="#1a1f30", zerolinecolor="#2a3050"),
    yaxis=dict(gridcolor="#1a1f30", linecolor="#1a1f30", zerolinecolor="#2a3050"),
)


# ── Tab 1: Route Analysis ──────────────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # Energy per segment bar chart
        seg_labels = [f"Seg {i+1}" for i in range(len(segs))]
        colors     = ["#00bfff" if v < 0 else "#00d4aa" for v in segs["total_wh"]]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=seg_labels,
            y=segs["total_wh"],
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Energy: %{y:.1f} Wh<extra></extra>",
        ))
        fig_bar.add_hline(y=0, line_color="#2a3050", line_width=1.5)
        fig_bar.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Energy per Segment (Wh)", font=dict(color="#e8eaf0", size=14)),
            showlegend=False,
            height=280,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        # Elevation profile
        cum_dist = np.cumsum([0] + segs["distance_km"].tolist())
        elevation = np.cumsum([0] + [
            row["distance_km"] * 1000 * np.sin(np.radians(row["slope_deg"]))
            for _, row in segs.iterrows()
        ])
        fig_elev = go.Figure()
        fig_elev.add_trace(go.Scatter(
            x=cum_dist, y=elevation,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0, 153, 255, 0.08)",
            line=dict(color="#0099ff", width=2),
            hovertemplate="Distance: %{x:.1f} km<br>Elevation: %{y:.0f} m<extra></extra>",
        ))
        fig_elev.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Elevation Profile (m)", font=dict(color="#e8eaf0", size=14)),
            height=280,
            xaxis_title="Distance (km)",
            yaxis_title="Elevation (m)",
        )
        st.plotly_chart(fig_elev, use_container_width=True)

    # Cumulative energy + consumption rate
    fig_cum = make_subplots(specs=[[{"secondary_y": True}]])
    cum_energy = np.cumsum([0] + segs["total_wh"].tolist())
    fig_cum.add_trace(go.Scatter(
        x=cum_dist, y=cum_energy,
        name="Cumulative Energy (Wh)",
        line=dict(color="#00d4aa", width=2.5),
        hovertemplate="Cumulative: %{y:.0f} Wh<extra></extra>",
    ), secondary_y=False)
    fig_cum.add_trace(go.Scatter(
        x=[(cum_dist[i] + cum_dist[i+1]) / 2 for i in range(len(segs))],
        y=segs["wh_km"],
        name="Consumption Rate (Wh/km)",
        mode="markers+lines",
        line=dict(color="#ffaa00", width=1.5, dash="dot"),
        marker=dict(size=7, color="#ffaa00"),
        hovertemplate="Rate: %{y:.0f} Wh/km<extra></extra>",
    ), secondary_y=True)
    fig_cum.add_hline(y=0, line_color="#2a3050", line_width=1)
    fig_cum.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Cumulative Energy & Consumption Rate", font=dict(color="#e8eaf0", size=14)),
        height=300,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899bb")),
    )
    fig_cum.update_yaxes(title_text="Cumulative Energy (Wh)", secondary_y=False,
                          gridcolor="#1a1f30", title_font=dict(color="#8899bb"))
    fig_cum.update_yaxes(title_text="Wh/km", secondary_y=True,
                          gridcolor="rgba(0,0,0,0)", title_font=dict(color="#ffaa00"))
    st.plotly_chart(fig_cum, use_container_width=True)


# ── Tab 2: Segment Breakdown ───────────────────────────────────────────────────
with tab2:
    # Surface composition pie
    col_a, col_b = st.columns([2, 3])

    with col_a:
        surface_counts = segs["surface"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=surface_counts.index,
            values=surface_counts.values,
            hole=0.55,
            marker=dict(colors=["#00d4aa", "#0099ff", "#ffaa00", "#ff6b6b", "#aa88ff"],
                        line=dict(color="#0d0f14", width=2)),
            textfont=dict(family="Space Mono", size=11),
            hovertemplate="%{label}<br>%{value} segments<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{len(segs)}</b><br><span style='font-size:10'>segs</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Space Mono", size=16, color="#e8eaf0")
        )
        pie_layout = {**PLOT_LAYOUT, "margin": dict(l=10, r=10, t=40, b=10)}
        fig_pie.update_layout(
            **pie_layout,
            title=dict(text="Surface Distribution", font=dict(color="#e8eaf0", size=14)),
            height=280,
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899bb", size=11)),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        # Scatter: slope vs wh_km, size=distance, color=surface
        surface_color_map = {
            "Smooth Asphalt": "#00d4aa",
            "Worn Asphalt":   "#0099ff",
            "Cobblestone":    "#ffaa00",
            "Gravel":         "#ff6b6b",
            "Dirt Road":      "#aa88ff",
        }
        fig_scatter = go.Figure()
        for surf, grp in segs.groupby("surface"):
            fig_scatter.add_trace(go.Scatter(
                x=grp["slope_deg"],
                y=grp["wh_km"],
                mode="markers",
                name=surf,
                marker=dict(
                    color=surface_color_map.get(surf, "#888"),
                    size=grp["distance_km"] * 4 + 8,
                    opacity=0.75,
                    line=dict(width=0),
                ),
                hovertemplate=f"<b>{surf}</b><br>Slope: %{{x:.1f}}°<br>Rate: %{{y:.0f}} Wh/km<br>Dist: %{{text}} km<extra></extra>",
                text=grp["distance_km"].round(1).astype(str),
            ))
        fig_scatter.add_hline(y=0, line_color="#2a3050", line_width=1.5,
                               annotation_text="regen threshold",
                               annotation_font=dict(color="#2a3050", size=10))
        fig_scatter.update_layout(
            **PLOT_LAYOUT,
            title=dict(text="Slope vs Consumption  (bubble size = distance)", font=dict(color="#e8eaf0", size=14)),
            xaxis_title="Slope (°)",
            yaxis_title="Wh/km",
            height=280,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899bb", size=11)),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Segment table
    st.markdown('<div class="section-header">SEGMENT LOG</div>', unsafe_allow_html=True)

    SLOPE_ICON = lambda s: "⬆️" if s > 1 else ("⬇️" if s < -1 else "➡️")
    SURF_EMOJI = lambda surf: SURFACE_SPECS.get(surf, {}).get("emoji", "")

    display_df = pd.DataFrame({
        "#":          range(1, len(segs) + 1),
        "Vehicle":    segs["vehicle"].str.split().str[1],
        "Surface":    segs["surface"].apply(lambda s: f"{SURF_EMOJI(s)} {s}"),
        "Dist (km)":  segs["distance_km"].round(2),
        "Slope":      segs["slope_deg"].apply(lambda s: f"{SLOPE_ICON(s)} {s:+.1f}°"),
        "Speed":      segs["speed_kmh"].apply(lambda v: f"{v} km/h"),
        "Bumps/km":   segs["bump_density"],
        "Temp":       segs["temperature_c"].apply(lambda t: f"{t:.0f}°C"),
        "Rate (Wh/km)": segs["wh_km"].round(0).astype(int),
        "Total (Wh)": segs["total_wh"].round(0).astype(int),
    })
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rate (Wh/km)": st.column_config.NumberColumn(format="%d"),
            "Total (Wh)":   st.column_config.NumberColumn(format="%d"),
        }
    )

    # Remove individual segment
    st.markdown("<br>", unsafe_allow_html=True)
    rem_col, _ = st.columns([1, 3])
    with rem_col:
        seg_to_remove = st.selectbox(
            "Remove segment",
            options=range(1, len(segs) + 1),
            format_func=lambda i: f"Segment {i}"
        )
    if st.button("🗑  Remove Selected"):
        st.session_state.segments.pop(seg_to_remove - 1)
        st.rerun()


# ── Tab 3: Physics Detail ──────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">ENERGY DECOMPOSITION</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    The model was trained on physics-derived labels using:<br>
    <code>E = (F_roll + F_gravity + F_drag − E_regen) × temp_factor</code><br><br>
    These components are computed below for each segment to cross-validate the model's output.
    </div>
    """, unsafe_allow_html=True)

    # Recompute physics components for visualization
    g = 9.81
    rho = 1.225

    physics_rows = []
    for _, seg in segs.iterrows():
        v    = VEHICLE_SPECS[seg["vehicle"]]
        s    = SURFACE_SPECS[seg["surface"]]
        d    = seg["distance_km"] * 1000
        spd  = seg["speed_kmh"] / 3.6
        sl   = np.radians(seg["slope_deg"])
        bump = seg["bump_density"]

        Crr      = s["Crr"] * (1 + (bump / 100) * 0.35)
        F_roll   = Crr * v["mass"] * g * np.cos(sl)
        F_grav   = v["mass"] * g * np.sin(sl)
        F_drag   = 0.5 * rho * v["Cd"] * v["A"] * spd**2
        regen    = v["regen"] * min(abs(seg["slope_deg"]) / 15, 1.0) if seg["slope_deg"] < -1 else 0
        E_regen  = regen * abs(F_grav) * d * v["regen"] / 3600

        physics_rows.append({
            "Roll":   F_roll * d / 3600,
            "Gravity":F_grav * d / 3600,
            "Drag":   F_drag * d / 3600,
            "Regen":  -E_regen,
        })

    phys_df = pd.DataFrame(physics_rows)

    # Stacked bar
    fig_stack = go.Figure()
    seg_labels = [f"Seg {i+1}" for i in range(len(segs))]
    colors_comp = {"Roll": "#0099ff", "Gravity": "#ff6b6b", "Drag": "#ffaa00", "Regen": "#00d4aa"}

    for comp in ["Roll", "Gravity", "Drag"]:
        fig_stack.add_trace(go.Bar(
            name=comp,
            x=seg_labels,
            y=phys_df[comp],
            marker_color=colors_comp[comp],
            hovertemplate=f"<b>{comp}</b><br>%{{y:.1f}} Wh<extra></extra>",
        ))
    fig_stack.add_trace(go.Bar(
        name="Regen",
        x=seg_labels,
        y=phys_df["Regen"],
        marker_color=colors_comp["Regen"],
        hovertemplate="<b>Regen</b><br>%{y:.1f} Wh<extra></extra>",
    ))
    fig_stack.update_layout(
        **PLOT_LAYOUT,
        barmode="relative",
        title=dict(text="Energy Breakdown by Physics Component (Wh)", font=dict(color="#e8eaf0", size=14)),
        height=340,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8899bb"),
                    orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # Summary table
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("""
        <div class="info-box">
        <b>Rolling Resistance</b> — increases with surface roughness (Crr) and bump density.
        Bump density adds up to +35% rolling resistance at 100 bumps/km.<br><br>
        <b>Gravity Component</b> — dominant on steep roads. Positive = uphill drain,
        negative = downhill recovery. Scales linearly with mass and slope.
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown("""
        <div class="info-box">
        <b>Aerodynamic Drag</b> — quadratic with speed. Doubles when speed increases
        from 60→85 km/h. Heavily influenced by frontal area and Cd.<br><br>
        <b>Regenerative Braking</b> — active on slopes below −1°, scales with steepness
        and vehicle regen efficiency (55%–75% depending on type).
        </div>
        """, unsafe_allow_html=True)

    if segs["temperature_c"].min() < 5:
        st.markdown("""
        <div class="warn-box">
        ⚠️ One or more segments have temperatures below 5°C. Battery efficiency drops
        significantly in cold weather — the temperature factor increases consumption
        by up to 15–20% below 0°C.
        </div>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#2a3050; font-family: Space Mono, monospace;
            font-size:11px; letter-spacing:2px; padding: 8px 0 24px 0;'>
  PHYSICS-INFORMED ML · XGBOOST · SYNTHETIC DATASET
</div>
""", unsafe_allow_html=True)
