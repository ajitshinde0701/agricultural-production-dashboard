import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# ---------------- CONFIG ----------------
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Agricultural Production Dashboard", layout="wide")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("cleaned_crop_production.csv")

# ---------------- TITLE ----------------
st.markdown(
    """
    <h1 style="text-align:center; color:#2E8B57;">
    🌾 Agricultural Production Analysis Dashboard
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

# ---- All States Totals ----
all_state_prod = df["production"].sum()
all_state_area = df["area"].sum()
all_state_yield = all_state_prod / all_state_area

st.markdown("### 🌍 Overall Agricultural Performance (All States)")
o1, o2, o3 = st.columns(3)
o1.metric("Total Production", f"{all_state_prod:,.0f}")
o2.metric("Total Area", f"{all_state_area:,.0f}")
o3.metric("Average Yield", f"{all_state_yield:.2f}")

# ---- Selected Region ----
region_prod = filtered_df["production"].sum()
region_area = filtered_df["area"].sum()
region_yield = region_prod / region_area if region_area > 0 else 0

prod_percent = (region_prod / all_state_prod) * 100 if all_state_prod > 0 else 0
area_percent = (region_area / all_state_area) * 100 if all_state_area > 0 else 0

if region_yield > all_state_yield:
    yield_status = "higher than"
elif region_yield < all_state_yield:
    yield_status = "lower than"
else:
    yield_status = "equal to"

st.markdown(f"### 📍 Selected Region ({state} - {district})")
r1, r2, r3 = st.columns(3)

r1.metric("Production", f"{region_prod:,.0f}", f"{prod_percent:.3f}% of all states")
r2.metric("Area", f"{region_area:,.0f}", f"{area_percent:.3f}% of all states")
r3.metric("Yield", f"{region_yield:.2f}", f"{yield_status} national avg")

st.info(
    f"""
    **Interpretation:**  
    - {district} contributes **{prod_percent:.3f}%** of total agricultural production across all states.  
    - The district uses **{area_percent:.3f}%** of total cultivated area.  
    - Its average yield is **{yield_status}** the overall national average yield.
    """
)

# ==================================================
# STATE-WISE TOTAL PRODUCTION COMPARISON
# ==================================================
st.subheader("🏛 State-wise Total Production Comparison")

state_prod = df.groupby("state")["production"].sum().sort_values(ascending=False)

fig_state, ax_state = plt.subplots(figsize=(10, 5))
colors = ["#2E8B57" if s == state else "#B0C4DE" for s in state_prod.index]

bars = ax_state.bar(state_prod.index, state_prod.values, color=colors)

ax_state.set_ylabel("Total Production")
ax_state.set_xlabel("State")
ax_state.set_title("State-wise Total Production Comparison")
ax_state.set_xticklabels(state_prod.index, rotation=90)

# ---- ADD VALUE LABELS ----
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax_state.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height/1e9:.1f}B",   # show in billions
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90
        )

st.pyplot(fig_state)




# STATE vs DISTRICT COMPARISON

st.subheader(f"📍 {state}: State vs District Comparison")

state_total_prod = state_year_df["production"].sum()
state_total_yield = state_year_df["production"].sum() / state_year_df["area"].sum()

colA, colB = st.columns(2)

with colA:
    fig_p, ax_p = plt.subplots()
    bars_p = ax_p.bar(
        ["State", "District"],
        [state_total_prod, region_prod],
        color=["#B0C4DE", "#2E8B57"]
    )
    ax_p.set_ylabel("Total Production")
    ax_p.set_title("Production Comparison")

    # Add value labels
    for bar in bars_p:
        height = bar.get_height()
        ax_p.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    st.pyplot(fig_p)

with colB:
    fig_y, ax_y = plt.subplots()
    bars_y = ax_y.bar(
        ["State", "District"],
        [state_total_yield, region_yield],
        color=["#B0C4DE", "#2E8B57"]
    )
    ax_y.set_ylabel("Average Yield")
    ax_y.set_title("Yield Comparison")

    # Add value labels
    for bar in bars_y:
        height = bar.get_height()
        ax_y.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    st.pyplot(fig_y)


# CROP-WISE PRODUCTION

st.subheader("🌾 Crop-wise Production Distribution")

crop_prod = filtered_df.groupby("crop")["production"].sum().sort_values(ascending=False)

fig_crop, ax_crop = plt.subplots(figsize=(8, 5))
ax_crop.barh(crop_prod.index, crop_prod.values)
ax_crop.set_xlabel("Production")

for i, v in enumerate(crop_prod.values):
    ax_crop.text(v, i, f" {int(v):,}", va="center")

st.pyplot(fig_crop)


# DATA TABLE

st.subheader("📋 Data")
st.dataframe(filtered_df)


