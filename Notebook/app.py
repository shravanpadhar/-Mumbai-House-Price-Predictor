import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mumbai House Price Predictor",
    page_icon="🏙️",
    layout="wide",
)


# ── Load model & metadata ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_meta():
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return meta


model = load_model()
meta = load_meta()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#D4691E;'>🏙️ Mumbai House Price Predictor</h1>
    <p style='text-align:center; color:gray; font-size:16px;'>
        Powered by a Random Forest model trained on 70,000+ Mumbai listings
    </p>
    <hr/>
""", unsafe_allow_html=True)

# ── Sidebar inputs ─────────────────────────────────────────────────────────────
st.sidebar.header("🏠 Property Details")
st.sidebar.markdown("Fill in the details below to get an estimated price.")

locality = st.sidebar.selectbox(
    "📍 Locality",
    options=meta["localities"],
    index=meta["localities"].index("Andheri") if "Andheri" in meta["localities"] else 0,
    help="Select the Mumbai locality / neighbourhood"
)

property_type = st.sidebar.selectbox(
    "🏢 Property Type",
    options=meta["property_types"],
    help="Type of the property"
)

bhk = st.sidebar.slider(
    "🛏️ BHK (Bedrooms)",
    min_value=meta["bhk_min"],
    max_value=meta["bhk_max"],
    value=2,
    step=1,
    help="Number of bedrooms"
)

area = st.sidebar.slider(
    "📐 Area (sq ft)",
    min_value=meta["area_min"],
    max_value=meta["area_max"],
    value=meta["area_median"],
    step=50,
    help="Carpet / built-up area in square feet"
)

# ── Prediction ─────────────────────────────────────────────────────────────────
input_df = pd.DataFrame([{
    "area": area,
    "locality": locality,
    "property_type": property_type,
    "bedroom_num": bhk,
}])

log_pred = model.predict(input_df)[0]
price_pred = np.expm1(log_pred)  # Reverse log1p transform
price_low = price_pred * 0.88  # ±12% confidence band
price_high = price_pred * 1.12


def fmt(n):
    """Format rupee values with Cr/L suffix."""
    if n >= 1e7:
        return f"₹{n / 1e7:.2f} Cr"
    elif n >= 1e5:
        return f"₹{n / 1e5:.2f} L"
    else:
        return f"₹{n:,.0f}"


ppsf = price_pred / area  # Price per sq ft

# ── Results layout ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="💰 Estimated Price",
        value=fmt(price_pred),
        help="Random Forest median prediction"
    )

with col2:
    st.metric(
        label="📉 Low Estimate",
        value=fmt(price_low),
        delta=f"-12% band",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="📈 High Estimate",
        value=fmt(price_high),
        delta=f"+12% band"
    )

st.markdown("---")

# ── Summary card ──────────────────────────────────────────────────────────────
st.subheader("📋 Your Property Summary")

sum_col1, sum_col2 = st.columns(2)

with sum_col1:
    st.markdown(f"""
    | Attribute        | Value               |
    |-----------------|---------------------|
    | Locality         | **{locality}**       |
    | Property Type    | **{property_type}**  |
    | BHK              | **{bhk} BHK**        |
    | Area             | **{area:,} sq ft**   |
    """)

with sum_col2:
    st.markdown(f"""
    | Estimate         | Value               |
    |-----------------|---------------------|
    | Predicted Price  | **{fmt(price_pred)}** |
    | Price per sq ft  | **₹{ppsf:,.0f}/sqft** |
    | Low Estimate     | **{fmt(price_low)}**  |
    | High Estimate    | **{fmt(price_high)}** |
    """)

# ── Price band visualisation (pure streamlit) ──────────────────────────────────
st.markdown("---")
st.subheader("📊 Price Range Visualisation")

chart_df = pd.DataFrame({
    "Estimate": ["Low (−12%)", "Predicted", "High (+12%)"],
    "Price (₹)": [price_low, price_pred, price_high],
})
st.bar_chart(chart_df.set_index("Estimate"))

# ── Sensitivity table ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 How Area Affects Price (same locality & BHK)")

areas_range = range(
    max(meta["area_min"], area - 500),
    min(meta["area_max"], area + 600),
    100
)
rows = []
for a in areas_range:
    inp = pd.DataFrame([{
        "area": a, "locality": locality,
        "property_type": property_type, "bedroom_num": bhk
    }])
    p = np.expm1(model.predict(inp)[0]) 
    rows.append({"Area (sq ft)": a, "Estimated Price": fmt(p), "Price/sqft": f"₹{p / a:,.0f}"})

st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr/>
<p style='text-align:center; color:gray; font-size:13px;'>
    Model: Random Forest Regressor &nbsp;|&nbsp; 
    Data: Mumbai real-estate listings &nbsp;|&nbsp;
    Predictions are estimates only — consult a real-estate professional for final valuations.
</p>
""", unsafe_allow_html=True)