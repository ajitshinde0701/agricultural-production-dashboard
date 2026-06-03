
# =========================================================
# AGRICULTURAL PRODUCTION ANALYSIS DASHBOARD
# LIGHTWEIGHT + PROFESSIONAL VERSION
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# =========================================================
# PAGE CONFIG
# =========================================================

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Agriculture Dashboard",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1 {
    color: #2E8B57;
}

.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

try:
    df = pd.read_csv("cleaned_crop_production.csv")

except Exception as e:

    st.error(f"Dataset Error: {e}")
    st.stop()

# =========================================================
# MODEL TRAINING
# =========================================================

@st.cache_resource
def train_model():

    data = df.copy()

    label_encoders = {}

    categorical_cols = [
        "state",
        "district",
        "season",
        "crop"
    ]

    # Encode
    for col in categorical_cols:

        le = LabelEncoder()

        data[col] = le.fit_transform(
            data[col].astype(str)
        )

        label_encoders[col] = le

    # Features
    X = data[
        [
            "state",
            "district",
            "season",
            "crop",
            "area",
            "year"
        ]
    ]

    # Target
    y = data["production"]

    # Lightweight Model
    model = LinearRegression()

    model.fit(X, y)

    return model, label_encoders


model, label_encoders = train_model()

# =========================================================
# TITLE
# =========================================================

st.markdown("""
<h1 style='text-align:center;'>
🌾 Agricultural Production Analysis Dashboard
</h1>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("📍 Dashboard Filters")

state = st.sidebar.selectbox(
    "Select State",
    sorted(df["state"].unique())
)

state_df = df[
    df["state"] == state
]

district = st.sidebar.selectbox(
    "Select District",
    sorted(state_df["district"].unique())
)

district_df = state_df[
    state_df["district"] == district
]

season = st.sidebar.selectbox(
    "Select Season",
    sorted(district_df["season"].unique())
)

season_df = district_df[
    district_df["season"] == season
]

crop_list = ["All Crops"] + sorted(
    season_df["crop"].unique()
)

crop = st.sidebar.selectbox(
    "Select Crop",
    crop_list
)

filtered_df = (
    season_df
    if crop == "All Crops"
    else season_df[
        season_df["crop"] == crop
    ]
)

# =========================================================
# YEAR FILTER
# =========================================================

min_year = int(df["year"].min())
max_year = int(df["year"].max())

year_range = st.sidebar.slider(
    "Select Year Range",
    min_year,
    max_year,
    (min_year, max_year)
)

filtered_df = filtered_df[
    (filtered_df["year"] >= year_range[0]) &
    (filtered_df["year"] <= year_range[1])
]

# =========================================================
# OVERVIEW METRICS
# =========================================================

st.subheader("📊 Dashboard Overview")

total_production = filtered_df["production"].sum()

total_area = filtered_df["area"].sum()

avg_yield = (
    total_production / total_area
    if total_area > 0 else 0
)

c1, c2, c3 = st.columns(3)

c1.metric(
    "🌾 Total Production",
    f"{total_production:,.0f}"
)

c2.metric(
    "🌍 Total Area",
    f"{total_area:,.0f}"
)

c3.metric(
    "📈 Average Yield",
    f"{avg_yield:.2f}"
)

# =========================================================
# LINE GRAPH
# =========================================================

st.subheader("📈 Production Trend")

trend_df = (
    filtered_df.groupby("year")["production"]
    .sum()
    .reset_index()
)

fig1, ax1 = plt.subplots(figsize=(10,5))

ax1.plot(
    trend_df["year"],
    trend_df["production"],
    marker='o',
    linewidth=3
)

ax1.set_title("Production Trend Over Years")

ax1.set_xlabel("Year")
ax1.set_ylabel("Production")

ax1.grid(True)

st.pyplot(fig1)

# =========================================================
# BAR CHART
# =========================================================

st.subheader("🌾 Top Crop Production")

crop_prod = (
    filtered_df.groupby("crop")["production"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

fig2, ax2 = plt.subplots(figsize=(10,5))

bars = ax2.bar(
    crop_prod.index,
    crop_prod.values
)

ax2.set_title("Top Crop Production")

ax2.set_xlabel("Crop")
ax2.set_ylabel("Production")

plt.xticks(rotation=45)

for bar in bars:

    height = bar.get_height()

    ax2.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:,.0f}",
        ha='center',
        va='bottom',
        fontsize=8
    )

st.pyplot(fig2)

# =========================================================
# PIE CHART
# =========================================================

st.subheader("🥧 Crop Distribution")

fig3, ax3 = plt.subplots(figsize=(7,7))

top_crop_share = crop_prod.head(5)

ax3.pie(
    top_crop_share.values,
    labels=top_crop_share.index,
    autopct='%1.1f%%'
)

ax3.set_title("Top Crop Share")

st.pyplot(fig3)

# =========================================================
# SCATTER PLOT
# =========================================================

st.subheader("🌾 Area vs Production")

fig4, ax4 = plt.subplots(figsize=(10,5))

ax4.scatter(
    filtered_df["area"],
    filtered_df["production"]
)

ax4.set_xlabel("Area")
ax4.set_ylabel("Production")

ax4.set_title("Area vs Production")

st.pyplot(fig4)

# =========================================================
# PREDICTION SECTION
# =========================================================

st.subheader("🤖 AI Production Prediction")

pc1, pc2 = st.columns(2)

with pc1:

    pred_state = st.selectbox(
        "Prediction State",
        sorted(df["state"].unique())
    )

    pred_district = st.selectbox(
        "Prediction District",
        sorted(
            df[
                df["state"] == pred_state
            ]["district"].unique()
        )
    )

    pred_season = st.selectbox(
        "Prediction Season",
        sorted(df["season"].unique())
    )

with pc2:

    pred_crop = st.selectbox(
        "Prediction Crop",
        sorted(df["crop"].unique())
    )

    pred_area = st.number_input(
        "Prediction Area",
        min_value=0.1,
        value=1.0
    )

    pred_year = st.number_input(
        "Prediction Year",
        min_value=2000,
        max_value=2050,
        value=2025
    )

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("🚀 Predict Production"):

    try:

        state_encoded = (
            label_encoders["state"]
            .transform([pred_state])[0]
        )

        district_encoded = (
            label_encoders["district"]
            .transform([pred_district])[0]
        )

        season_encoded = (
            label_encoders["season"]
            .transform([pred_season])[0]
        )

        crop_encoded = (
            label_encoders["crop"]
            .transform([pred_crop])[0]
        )

        input_df = pd.DataFrame({

            "state": [state_encoded],
            "district": [district_encoded],
            "season": [season_encoded],
            "crop": [crop_encoded],
            "area": [pred_area],
            "year": [pred_year]

        })

        prediction = model.predict(input_df)[0]

        predicted_yield = (
            prediction / pred_area
        )

        st.success(
            "Prediction Successful ✅"
        )

        p1, p2 = st.columns(2)

        p1.metric(
            "🌾 Predicted Production",
            f"{prediction:,.2f}"
        )

        p2.metric(
            "📈 Predicted Yield",
            f"{predicted_yield:.2f}"
        )

        # =================================================
        # FUTURE FORECAST
        # =================================================

        st.subheader("📉 Future Forecast")

        future_years = list(
            range(pred_year, pred_year + 6)
        )

        future_predictions = []

        for yr in future_years:

            future_input = pd.DataFrame({

                "state": [state_encoded],
                "district": [district_encoded],
                "season": [season_encoded],
                "crop": [crop_encoded],
                "area": [pred_area],
                "year": [yr]

            })

            future_pred = model.predict(
                future_input
            )[0]

            future_predictions.append(
                future_pred
            )

        future_df = pd.DataFrame({

            "Year": future_years,
            "Production": future_predictions

        })

        fig5, ax5 = plt.subplots(figsize=(10,5))

        ax5.plot(
            future_df["Year"],
            future_df["Production"],
            marker='o',
            linewidth=3
        )

        ax5.set_title(
            "Future Production Forecast"
        )

        ax5.set_xlabel("Year")
        ax5.set_ylabel("Production")

        ax5.grid(True)

        st.pyplot(fig5)

        # =================================================
        # HISTORICAL vs PREDICTED
        # =================================================

        st.subheader("📈 Historical vs Predicted")

        historical_df = df[
            (df["state"] == pred_state) &
            (df["district"] == pred_district) &
            (df["crop"] == pred_crop)
        ]

        yearly_prod = (
            historical_df.groupby("year")
            ["production"]
            .mean()
            .reset_index()
        )

        fig6, ax6 = plt.subplots(figsize=(10,5))

        ax6.plot(
            yearly_prod["year"],
            yearly_prod["production"],
            marker='o',
            linewidth=2,
            label="Historical"
        )

        ax6.scatter(
            pred_year,
            prediction,
            s=250,
            marker='*',
            label="Prediction"
        )

        ax6.legend()

        ax6.set_title(
            "Historical vs Predicted Production"
        )

        ax6.set_xlabel("Year")
        ax6.set_ylabel("Production")

        ax6.grid(True)

        st.pyplot(fig6)

        # =================================================
        # SMART SUGGESTIONS
        # =================================================

        st.subheader("🌱 Farming Suggestions")

        if predicted_yield > avg_yield:

            st.success("""
            ✅ Excellent yield expected.

            Suggestions:
            - Maintain irrigation
            - Continue balanced fertilizer
            - Monitor pest attacks
            """)

        else:

            st.warning("""
            ⚠ Lower yield expected.

            Suggestions:
            - Improve irrigation
            - Use organic compost
            - Apply fertilizers properly
            """)

    except Exception as e:

        st.error(f"Prediction Error: {e}")

# =========================================================
# DATA TABLE
# =========================================================

st.subheader("📋 Filtered Dataset")

st.dataframe(filtered_df)

# =========================================================
# DOWNLOAD BUTTON
# =========================================================

csv = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "📥 Download Dataset",
    csv,
    "filtered_dataset.csv",
    "text/csv"
)

