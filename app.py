# =========================================================
# LIGHTWEIGHT AGRICULTURE DASHBOARD
# STREAMLIT CLOUD FRIENDLY
# FIXED VERSION (MISSING STATES + STREAMLIT WARNINGS FIXED)
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# =========================================================
# OPTIONAL PLOTLY SUPPORT
# =========================================================

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

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

@st.cache_data
def load_data():

    df = pd.read_csv("cleaned_crop_production.csv")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Clean text columns
    for col in ["state", "district", "season", "crop"]:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.title()
        )

    # Convert numeric columns safely
    df["production"] = pd.to_numeric(
        df["production"],
        errors="coerce"
    )

    df["area"] = pd.to_numeric(
        df["area"],
        errors="coerce"
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    # Drop only important nulls
    df = df.dropna(
        subset=[
            "state",
            "district",
            "season",
            "crop",
            "production",
            "area",
            "year"
        ]
    )

    return df


try:

    df = load_data()

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

    # Encode categorical columns
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

    # Model
    model = DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    )

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
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("📍 Dashboard Filters")

state = st.sidebar.selectbox(
    "Select State",
    sorted(df["state"].dropna().unique())
)

state_df = df[
    df["state"] == state
]

district = st.sidebar.selectbox(
    "Select District",
    sorted(state_df["district"].dropna().unique())
)

district_df = state_df[
    state_df["district"] == district
]

season = st.sidebar.selectbox(
    "Select Season",
    sorted(district_df["season"].dropna().unique())
)

season_df = district_df[
    district_df["season"] == season
]

crop_list = ["All Crops"] + sorted(
    season_df["crop"].dropna().unique()
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
# YEAR RANGE
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

if PLOTLY_AVAILABLE:

    fig1 = px.line(
        trend_df,
        x="year",
        y="production",
        markers=True,
        title="Production Trend"
    )

    st.plotly_chart(
        fig1,
        width="stretch"
    )

else:

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(
        trend_df["year"],
        trend_df["production"],
        marker='o',
        linewidth=3
    )

    ax1.set_title("Production Trend")

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

if PLOTLY_AVAILABLE:

    crop_df = crop_prod.reset_index()

    fig2 = px.bar(
        crop_df,
        x="crop",
        y="production",
        color="production",
        title="Top Crop Production"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )

else:

    fig2, ax2 = plt.subplots(figsize=(10, 5))

    ax2.bar(
        crop_prod.index,
        crop_prod.values
    )

    ax2.set_title("Top Crop Production")

    ax2.set_xlabel("Crop")
    ax2.set_ylabel("Production")

    plt.xticks(rotation=45)

    st.pyplot(fig2)

# =========================================================
# STATE COMPARISON
# =========================================================

st.subheader("🏛 State-wise Production")

state_prod = (
    df.groupby("state")["production"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

if PLOTLY_AVAILABLE:

    state_df_plot = state_prod.reset_index()

    fig3 = px.bar(
        state_df_plot,
        x="state",
        y="production",
        color="production",
        title="Top Producing States"
    )

    st.plotly_chart(
        fig3,
        width="stretch"
    )

else:

    fig3, ax3 = plt.subplots(figsize=(12, 5))

    ax3.bar(
        state_prod.index,
        state_prod.values
    )

    ax3.set_title("Top Producing States")

    ax3.set_xlabel("State")
    ax3.set_ylabel("Production")

    plt.xticks(rotation=45)

    st.pyplot(fig3)

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
# PREDICT BUTTON
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

        predicted_yield = prediction / pred_area

        st.success("Prediction Successful ✅")

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
        # FUTURE FORECAST GRAPH
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

        if PLOTLY_AVAILABLE:

            fig4 = px.line(
                future_df,
                x="Year",
                y="Production",
                markers=True,
                title="Future Production Forecast"
            )

            st.plotly_chart(
                fig4,
                width="stretch"
            )

        else:

            fig4, ax4 = plt.subplots(figsize=(10, 5))

            ax4.plot(
                future_df["Year"],
                future_df["Production"],
                marker='o',
                linewidth=3
            )

            ax4.grid(True)

            st.pyplot(fig4)

        # =================================================
        # HISTORICAL VS PREDICTION
        # =================================================

        st.subheader("📈 Historical vs Prediction")

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

        if PLOTLY_AVAILABLE:

            fig5 = px.line(
                yearly_prod,
                x="year",
                y="production",
                markers=True,
                title="Historical Production"
            )

            fig5.add_scatter(
                x=[pred_year],
                y=[prediction],
                mode="markers",
                marker=dict(size=15),
                name="Prediction"
            )

            st.plotly_chart(
                fig5,
                width="stretch"
            )

        else:

            fig5, ax5 = plt.subplots(figsize=(10, 5))

            ax5.plot(
                yearly_prod["year"],
                yearly_prod["production"],
                marker='o',
                linewidth=2,
                label="Historical"
            )

            ax5.scatter(
                pred_year,
                prediction,
                s=250,
                marker='*',
                label="Prediction"
            )

            ax5.legend()

            ax5.grid(True)

            st.pyplot(fig5)

        # =================================================
        # SUGGESTIONS
        # =================================================

        st.subheader("🌱 Smart Farming Suggestions")

        if predicted_yield > avg_yield:

            st.success("""
            ✅ Excellent yield expected.

            Suggestions:
            - Maintain irrigation
            - Use balanced fertilizer
            - Continue pest monitoring
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

st.subheader("📋 Dataset Preview")

st.dataframe(filtered_df.head(100))

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