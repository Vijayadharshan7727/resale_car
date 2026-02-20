import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="🚗 Car Value AI",
    page_icon="🚗",
    layout="wide"
)

# ---------------- DARK AI THEME ---------------- #
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0E1117, #111827);
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg,#00C2FF,#007BFF);
    color: white;
    border-radius: 8px;
    font-weight: bold;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #
st.markdown("<h1 style='text-align:center;'>🚗 Used Car Resale Value AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Linear • Ridge • Lasso Regression</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("cardekho_dataset.csv")  # ⚠️ Change to your actual file name
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

# ---------------- AUTO DETECT TARGET ---------------- #
if "selling_price" in df.columns:
    target_column = "selling_price"
elif "price" in df.columns:
    target_column = "price"
else:
    st.error("❌ Target column not found. Rename it to 'selling_price' or 'price'.")
    st.write("Available columns:", df.columns)
    st.stop()

# ---------------- FEATURE ENGINEERING ---------------- #
if "year" in df.columns:
    df["car_age"] = 2025 - df["year"]
    df.drop("year", axis=1, inplace=True)

# Remove rows with missing target
df = df.dropna(subset=[target_column])

X = df.drop(target_column, axis=1)
y = df[target_column]

# Identify column types
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# ---------------- SAFE PREPROCESSOR ---------------- #
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
])

# ---------------- SIDEBAR ---------------- #
st.sidebar.header("⚙️ Car Configuration")

input_data = {}

for col in num_cols:
    input_data[col] = st.sidebar.slider(
        col,
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].mean())
    )

for col in cat_cols:
    input_data[col] = st.sidebar.selectbox(
        col,
        df[col].dropna().unique()
    )

model_choice = st.sidebar.radio(
    "Select ML Model",
    ["Linear Regression", "Ridge", "Lasso"]
)

predict_button = st.sidebar.button("🚀 Predict Price")

# ---------------- MODEL SELECTION ---------------- #
if model_choice == "Linear Regression":
    model = LinearRegression()
elif model_choice == "Ridge":
    model = Ridge(alpha=1.0)
else:
    model = Lasso(alpha=0.1)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])

# ---------------- TRAIN MODEL ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred_test = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# ---------------- DASHBOARD LAYOUT ---------------- #
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("### 📊 Model Performance")
    st.metric("R² Score", round(r2, 3))
    st.metric("RMSE", round(rmse, 3))
    st.markdown(f"Model: **{model_choice}**")

# ---------------- PREDICTION ---------------- #
if predict_button:
    input_df = pd.DataFrame([input_data])
    prediction = pipeline.predict(input_df)[0]

    with col1:
        st.markdown("## 💰 Predicted Resale Value")
        st.markdown(
            f"<h2 style='color:#00C2FF;'>₹ {round(prediction,2)}</h2>",
            unsafe_allow_html=True
        )
        st.success("Prediction Generated Successfully 🚀")

# ---------------- HISTOGRAM ---------------- #
st.markdown("### 📈 Market Price Distribution")

fig = px.histogram(
    df,
    x=target_column,
    nbins=40,
    template="plotly_dark",
    title="Used Car Price Distribution"
)

if predict_button:
    fig.add_vline(
        x=prediction,
        line_dash="dash",
        line_color="red",
        annotation_text="Predicted Price"
    )

st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>Built by Vijayadharshan | AI & Data Science Portfolio Project</center>",
    unsafe_allow_html=True
)
