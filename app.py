import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("uber.csv")

df = load_data()

st.title("🚖 Uber Ride Analytics & Prediction Dashboard")

# -------------------------------
# Data Preview
# -------------------------------
st.subheader("📊 Dataset Overview")
st.write(df.head())

# -------------------------------
# ML Section
# -------------------------------
st.subheader("🤖 ML Model Training & Prediction")

target_col = "Booking Status"
if target_col in df.columns:

    # Features + target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode categoricals
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le

    # Ensure numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
    }

    model_choice = st.selectbox("Select Model", list(models.keys()))
    model = models[model_choice]

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds) * 100
    st.success(f"✅ {model_choice} Accuracy: {acc:.2f}%")

    # -------------------------------
    # User-Friendly Prediction Form
    # -------------------------------
    st.write("### 🎯 Try a Prediction")

    input_data = {}

    # Vehicle Type
    if "Vehicle Type" in df.columns:
        input_data["Vehicle Type"] = st.selectbox(
            "Select Vehicle Type", df["Vehicle Type"].dropna().unique()
        )

    # Pickup Location
    if "Pickup Location" in df.columns:
        input_data["Pickup Location"] = st.selectbox(
            "Select Pickup Location", df["Pickup Location"].dropna().unique()
        )

    # Drop Location
    if "Drop Location" in df.columns:
        input_data["Drop Location"] = st.selectbox(
            "Select Drop Location", df["Drop Location"].dropna().unique()
        )

    # Payment Method
    if "Payment Method" in df.columns:
        input_data["Payment Method"] = st.radio(
            "Select Payment Method", df["Payment Method"].dropna().unique()
        )

    # Ride Distance
    if "Ride Distance" in df.columns:
        min_val = int(df["Ride Distance"].min())
        max_val = int(df["Ride Distance"].max())
        input_data["Ride Distance"] = st.slider(
            "Select Ride Distance (km)", min_val, max_val, step=1
        )

    # Booking Value
    if "Booking Value" in df.columns:
        min_val = int(df["Booking Value"].min())
        max_val = int(df["Booking Value"].max())
        input_data["Booking Value"] = st.slider(
            "Select Booking Value (₹)", min_val, max_val, step=50
        )

    # -------------------------------
    # Run Prediction
    # -------------------------------
    if st.button("🔮 Predict Booking Status"):
        input_df = pd.DataFrame([input_data])

        # Apply encoders
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        st.success(f"🚕 Predicted Booking Status: **{prediction}**")

    # -------------------------------
    # Feature Importance
    # -------------------------------
    if model_choice in ["Decision Tree", "Random Forest"]:
        st.write("### 🔑 Feature Importance")
        feature_importances = pd.Series(
            model.feature_importances_, index=df.drop(columns=[target_col]).columns
        )
        fig, ax = plt.subplots()
        feature_importances.sort_values(ascending=False).head(10).plot(kind="bar", ax=ax)
        st.pyplot(fig)

else:
    st.error("❌ 'Booking Status' column not found in dataset.")
