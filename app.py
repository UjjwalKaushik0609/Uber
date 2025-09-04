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
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="Uber Ride Analysis", layout="wide")
st.title("🚕 Uber Ride Analytics & Prediction Dashboard")

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your Uber CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    target_col = "Booking Status"
    if target_col in df.columns:
        # -------------------------------
        # Preprocessing
        # -------------------------------
        X = df.drop(columns=[target_col])
        y = df[target_col]

        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------------------------------
        # Models with Class Balancing
        # -------------------------------
        class_weights = "balanced"

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, class_weight=class_weights),
            "Decision Tree": DecisionTreeClassifier(class_weight=class_weights),
            "Random Forest": RandomForestClassifier(class_weight=class_weights),
            "Naive Bayes": GaussianNB(),  # NB doesn't support class_weight
            "SVM": SVC(class_weight=class_weights),
            "KNN": KNeighborsClassifier(),  # KNN doesn't support class_weight
        }

        model_choice = st.selectbox("Select Model", list(models.keys()))
        model = models[model_choice]
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100
        st.success(f"✅ {model_choice} Accuracy: {acc:.2f}%")

        # -------------------------------
        # Confusion Matrix & Report
        # -------------------------------
        st.write("### 📉 Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.write("### 📑 Classification Report")
        st.text(classification_report(y_test, preds))

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
        # Run Prediction (FIXED)
        # -------------------------------
        if st.button("🔮 Predict Booking Status"):
            input_df = pd.DataFrame([input_data])

            # Apply label encoders where needed
            for col in input_df.columns:
                if col in label_encoders:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

            # ✅ Rebuild full feature set (same columns as training)
            full_input = pd.DataFrame(columns=df.drop(columns=[target_col]).columns)
            for col in full_input.columns:
                if col in input_df.columns:
                    full_input.at[0, col] = input_df[col].values[0]
                else:
                    full_input.at[0, col] = 0  # default for missing

            full_input = full_input.apply(pd.to_numeric, errors="coerce").fillna(0)

            # Scale using trained scaler
            input_scaled = scaler.transform(full_input)

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
else:
    st.info("👆 Please upload a dataset to continue.")
