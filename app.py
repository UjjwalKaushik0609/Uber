import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --------------------------------
# Page Config
# --------------------------------
st.set_page_config(page_title="Uber Data Analytics Dashboard", layout="wide")

st.title("🚖 Uber Data Analytics Dashboard & Prediction")
st.markdown("Upload your Uber ride bookings dataset to explore trends and predict booking outcomes.")

# --------------------------------
# Upload Data
# --------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data uploaded successfully!")
    st.write("### Data Preview", df.head())

    # --------------------------------
    # EDA
    # --------------------------------
    st.subheader("📊 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Booking Status Distribution**")
        fig, ax = plt.subplots()
        sns.countplot(x="Booking Status", data=df, palette="viridis", ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("**Vehicle Type Distribution**")
        fig, ax = plt.subplots()
        sns.countplot(x="Vehicle Type", data=df, order=df['Vehicle Type'].value_counts().index, palette="coolwarm", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --------------------------------
    # Preprocessing
    # --------------------------------
    st.subheader("⚙️ Preprocessing & Feature Engineering")

    # Handle Date/Time
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df = df.drop(columns=["Date"])

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df["Hour"] = df["Time"].dt.hour
        df["Minute"] = df["Time"].dt.minute
        df = df.drop(columns=["Time"])

    # Drop leakage cols if exist
    leakage_cols = [
        "Reason for cancelling by Customer", "Driver Cancellation Reason",
        "Customer Rating", "Driver Ratings",
        "Cancelled Rides by Customer", "Cancelled Rides by Driver",
        "Incomplete Rides", "Incomplete Rides Reason"
    ]
    df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")

    # Encode categoricals
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object' or isinstance(df[col].iloc[0], str):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Features & target
    target_col = "Booking Status"
    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' not found in dataset.")
        st.stop()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --------------------------------
    # Train Models
    # --------------------------------
    st.subheader("🤖 Model Training & Evaluation")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    # Try adding XGBoost
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(eval_metric='mlogloss')
    except ImportError:
        st.warning("⚠️ XGBoost not installed, skipping.")

    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        accuracies[name] = round(acc, 2)

    results_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy (%)"]).sort_values(by="Accuracy (%)", ascending=False)

    st.write("### Model Accuracies", results_df)

    fig, ax = plt.subplots()
    sns.barplot(x="Accuracy (%)", y="Model", data=results_df, palette="viridis", ax=ax)
    st.pyplot(fig)

    # --------------------------------
    # Prediction
    # --------------------------------
    st.subheader("🔮 Predict Booking Status")

    sample_input = {}
    for col in df.drop(columns=[target_col]).columns:
        min_val = int(df[col].min())
        max_val = int(df[col].max())
        default_val = int(df[col].median())
        sample_input[col] = st.slider(f"{col}", min_val, max_val, default_val)

    if st.button("Predict"):
        input_df = pd.DataFrame([sample_input])
        input_scaled = scaler.transform(input_df)

        best_model_name = results_df.iloc[0]["Model"]
        best_model = models[best_model_name]

        prediction = best_model.predict(input_scaled)[0]
        st.success(f"✅ Predicted Booking Status: **{prediction}** (using {best_model_name})")
else:
    st.info("👆 Upload your `ncr_ride_bookings.csv` file to start analysis.")
