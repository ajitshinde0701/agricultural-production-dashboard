import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings

# ---------------- CONFIG ----------------
warnings.filterwarnings("ignore")
rcParams["font.family"] = "Noto Sans Devanagari"
rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="Agricultural Production Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("cleaned_crop_production.csv")

# ---------------- CROP NAME MAP ----------------
crop_map = {
    "Rice": "भात", "Wheat": "गहू", "Maize": "मका",
    "Sugarcane": "ऊस", "Cotton": "कापूस",
    "Jowar": "ज्वारी", "Bajra": "बाजरी",
    "Pulses": "डाळी", "Groundnut": "भुईमूग",
    "Soybean": "सोयाबीन"
}
df["crop_mr"] = df["crop"].map(crop_map).fillna(df["crop"])

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style="text-align:center; color:#2E8B57;">
    🌾 Agricultural Production Analysis Dashboard<br>
    <span style="font-size:18px;">कृषी उत्पादन विश्लेषण डॅशबोर्ड</span>
    </h1>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("📍 Filters")

state = st.sidebar.selectbox("State", sorted(df["state"].unique()))
state_df = df[df["state"] == state]

district = st.sidebar.selectbox("District", sorted(state_df["district"].unique()))
district_df = state_df[state_df["district"] == district]

season = st.sidebar.selectbox("Season", sorted(district_df["season"].unique()))
season_df = district_df[district_df["season"] == season]

crop_list = ["All Crops"] + sorted(season_df["crop"].unique())
crop = st.sidebar.selectbox("Crop", crop_list)

filtered_df = season_df if crop == "All Crops" else season_df[season_df["crop"] == crop]

# ---------------- YEAR RANGE ----------------
min_year, max_year = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

filtered_df = filtered_df[
    (filtered_df["year"] >= year_range[0]) &
    (filtered_df["year"] <= year_range[1])
]

state_year_df = state_df[
    (state_df["year"] >= year_range[0]) &
    (state_df["year"] <= year_range[1])
]

# ==================================================
# OVERVIEW (2 ROWS)
# ==================================================
st.subheader("📊 Overview")

# ---- Row 1: Overall ----
overall_prod = df["production"].sum()
overall_area = df["area"].sum()
overall_yield = overall_prod / overall_area

st.markdown("### 🌍 Overall Agricultural Performance")
o1, o2, o3 = st.columns(3)
o1.metric("Total Production", f"{overall_prod:,.0f}")
o2.metric("Total Area", f"{overall_area:,.0f}")
o3.metric("Average Yield", f"{overall_yield:.2f}")

# ---- Row 2: Selected Region ----
region_prod = filtered_df["production"].sum()
region_area = filtered_df["area"].sum()
region_yield = region_prod / region_area if region_area > 0 else 0

st.markdown(f"### 📍 Selected Region ({state} - {district})")
r1, r2, r3 = st.columns(3)
r1.metric("Production", f"{region_prod:,.0f}")
r2.metric("Area", f"{region_area:,.0f}")
r3.metric("Yield", f"{region_yield:.2f}")

# ==================================================
# STATE-WISE TOTAL PRODUCTION COMPARISON
# ==================================================
st.subheader("🏛 State-wise Total Production Comparison")

state_prod = (
    df.groupby("state")["production"]
    .sum()
    .sort_values(ascending=False)
)

fig_state, ax_state = plt.subplots(figsize=(9, 5))
colors = ["#2E8B57" if s == state else "#B0C4DE" for s in state_prod.index]

ax_state.bar(state_prod.index, state_prod.values, color=colors)
ax_state.set_ylabel("Total Production")
ax_state.set_xlabel("State")
ax_state.set_xticklabels(state_prod.index, rotation=90)

st.pyplot(fig_state)
st.caption("Selected state is highlighted for comparison.")

# ==================================================
# STATE vs DISTRICT COMPARISON (BAR + TEXT)
# ==================================================
st.subheader(f"📍 {state}: State vs District Comparison")

# ---- Production ----
state_total_prod = state_year_df["production"].sum()
district_total_prod = filtered_df["production"].sum()

# ---- Yield ----
state_yield = state_year_df["production"].sum() / state_year_df["area"].sum()
district_yield = region_yield

colA, colB = st.columns(2)

with colA:
    st.markdown("#### 🌾 Total Production")
    fig_p, ax_p = plt.subplots()
    ax_p.bar(
        ["State", "District"],
        [state_total_prod, district_total_prod],
        color=["#2E8B57", "#FFA07A"]
    )
    ax_p.set_ylabel("Production")
    st.pyplot(fig_p)

with colB:
    st.markdown("#### 🌱 Average Yield")
    fig_y, ax_y = plt.subplots()
    ax_y.bar(
        ["State", "District"],
        [state_yield, district_yield],
        color=["#2E8B57", "#FFA07A"]
    )
    ax_y.set_ylabel("Yield")
    st.pyplot(fig_y)

# ---------------- TEXTUAL INTERPRETATION ----------------
st.markdown("### 📝 Interpretation (State vs District)")

prod_share = (district_total_prod / state_total_prod) * 100 if state_total_prod > 0 else 0

if district_yield > state_yield:
    yield_compare = "higher than"
elif district_yield < state_yield:
    yield_compare = "lower than"
else:
    yield_compare = "equal to"

st.success(
    f"""
    • **{district} district contributes {prod_share:.2f}% of the total agricultural production of {state}.**  
    • **Average yield of {district} is {yield_compare} the state average yield.**  
    • This shows the relative importance and productivity of the district within the state.
    """
)

# ==================================================
# CROP-WISE PRODUCTION
# ==================================================
st.subheader("🌾 Crop-wise Production Distribution")

crop_prod = (
    filtered_df.groupby("crop_mr")["production"]
    .sum()
    .sort_values(ascending=False)
)

fig_crop, ax_crop = plt.subplots(figsize=(8, 5))
ax_crop.barh(crop_prod.index, crop_prod.values)
ax_crop.set_xlabel("Production")

for i, v in enumerate(crop_prod.values):
    ax_crop.text(v, i, f" {int(v):,}", va="center")

st.pyplot(fig_crop)

# ---- Interpretation ----
if not crop_prod.empty:
    share = (crop_prod.iloc[0] / crop_prod.sum()) * 100
    if share > 70:
        st.warning(
            f"🔎 Interpretation: Production is dominated by **{crop_prod.index[0]}** "
            f"({share:.1f}% contribution)."
        )
    else:
        st.info("🔎 Interpretation: Production is distributed among multiple crops.")

# ==================================================
# DATA TABLE
# ==================================================
st.subheader("📋 Data")
st.dataframe(filtered_df)

# ---------------- FOOTER ----------------
st.caption(
    "This dashboard compares overall, state-level, and district-level agricultural performance "
    "using both visual and textual analysis. All analysis is descriptive."
)
