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

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="Uber Ride Bookings Analysis", layout="wide")
st.title("🚕 Uber Ride Bookings Dashboard & ML Prediction")

# -------------------------------
# Upload Dataset
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Data Cleaning
    # -------------------------------
    st.subheader("🧹 Data Cleaning")
    df.drop_duplicates(inplace=True)
    df.fillna("Unknown", inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day

    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        df["Hour"] = df["Time"].dt.hour
        df["Minute"] = df["Time"].dt.minute

    leakage_cols = [
        "Reason for cancelling by Customer",
        "Driver Cancellation Reason",
        "Customer Rating",
        "Driver Ratings",
        "Cancelled Rides by Customer",
        "Cancelled Rides by Driver",
        "Incomplete Rides",
        "Incomplete Rides Reason"
    ]
    df.drop(columns=[col for col in leakage_cols if col in df.columns],
            inplace=True, errors="ignore")

    st.success("✅ Data cleaned successfully!")

    # -------------------------------
    # Download Cleaned Dataset
    # -------------------------------
    st.write("📥 Download Cleaned Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=csv,
        file_name="cleaned_ride_bookings.csv",
        mime="text/csv",
    )

    # -------------------------------
    # Sidebar Filters
    # -------------------------------
    st.sidebar.header("🔎 Filters")

    if "Date" in df.columns:
        min_date, max_date = df["Date"].min(), df["Date"].max()
        date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
        if len(date_range) == 2:
            df = df[(df["Date"] >= pd.to_datetime(date_range[0])) &
                    (df["Date"] <= pd.to_datetime(date_range[1]))]

    if "Vehicle Type" in df.columns:
        vehicle_filter = st.sidebar.multiselect(
            "Select Vehicle Types",
            df["Vehicle Type"].unique(),
            default=df["Vehicle Type"].unique()
        )
        df = df[df["Vehicle Type"].isin(vehicle_filter)]

    if "Booking Status" in df.columns:
        status_filter = st.sidebar.multiselect(
            "Select Booking Status",
            df["Booking Status"].unique(),
            default=df["Booking Status"].unique()
        )
        df = df[df["Booking Status"].isin(status_filter)]

    # -------------------------------
    # EDA Visualizations
    # -------------------------------
    st.subheader("📈 Exploratory Data Analysis")

    if "Date" in df.columns and "Booking Value" in df.columns:
        st.write("### Daily Rides & Revenue")
        df["Booking Value"] = pd.to_numeric(df["Booking Value"], errors="coerce").fillna(0)
        rides_per_day = df.groupby("Date").size().reset_index(name="Total Rides")
        revenue_per_day = df.groupby("Date")["Booking Value"].sum().reset_index(name="Total Revenue")
        daily_stats = pd.merge(rides_per_day, revenue_per_day, on="Date")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily_stats["Date"], daily_stats["Total Rides"], label="Total Rides", color="blue")
        ax.plot(daily_stats["Date"], daily_stats["Total Revenue"], label="Total Revenue", color="green")
        ax.legend()
        st.pyplot(fig)

    if "Booking Status" in df.columns:
        st.write("### Booking Status Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Booking Status", data=df, ax=ax, palette="viridis")
        st.pyplot(fig)

    if "Vehicle Type" in df.columns:
        st.write("### Vehicle Type Distribution")
        fig, ax = plt.subplots()
        sns.countplot(
            x="Vehicle Type",
            data=df,
            order=df["Vehicle Type"].value_counts().index,
            palette="coolwarm"
        )
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if "Pickup Location" in df.columns and "Drop Location" in df.columns:
        st.write("### Top 10 Pickup & Drop Locations")
        top_pickups = df["Pickup Location"].value_counts().head(10)
        top_drops = df["Drop Location"].value_counts().head(10)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(x=top_pickups.values, y=top_pickups.index, ax=axes[0], palette="Blues_r")
        sns.barplot(x=top_drops.values, y=top_drops.index, ax=axes[1], palette="Greens_r")
        axes[0].set_title("Pickup Locations")
        axes[1].set_title("Drop Locations")
        st.pyplot(fig)

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # -------------------------------
    # ML Prediction Section (Updated)
    # -------------------------------
    st.subheader("🤖 Predict Booking Status")
    target_col = "Booking Status"

    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        datetime_cols = X.select_dtypes(include=["datetime64[ns]"]).columns
        X = X.drop(columns=datetime_cols, errors="ignore")
        y = df[target_col]

        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

        if len(np.unique(y)) < 2:
            st.warning("⚠️ Not enough classes in the selected filters for training. Please select more Booking Status options.")
        else:
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

            if X.shape[0] > 0:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )

                models = {
                    "Logistic Regression": LogisticRegression(max_iter=500),
                    "Decision Tree": DecisionTreeClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "Naive Bayes": GaussianNB(),
                    "SVM": SVC(),
                    "KNN": KNeighborsClassifier()
                }

                model_choice = st.selectbox("Select Model", list(models.keys()))
                model = models[model_choice]

                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds) * 100
                st.success(f"✅ {model_choice} Accuracy: {acc:.2f}%")

                if model_choice in ["Decision Tree", "Random Forest"]:
                    st.write("### 🔎 Feature Importance")
                    feature_importances = pd.Series(
                        model.feature_importances_,
                        index=df.drop(columns=[target_col] + list(datetime_cols), errors="ignore").columns
                    )
                    fig, ax = plt.subplots()
                    feature_importances.sort_values(ascending=False).head(10).plot(kind="bar", ax=ax)
                    st.pyplot(fig)

                st.write("### 🎯 Make a Prediction")
                with st.form("prediction_form"):
                    booking_value = st.number_input("Booking Value", min_value=0.0, step=1.0)
                    if "Vehicle Type" in df.columns:
                        vehicle_type = st.selectbox("Vehicle Type", df["Vehicle Type"].unique())
                    else:
                        vehicle_type = "Unknown"
                    hour = st.slider("Hour of Day", 0, 23, 12)
                    submit = st.form_submit_button("Predict")

                if submit:
                    input_df = pd.DataFrame({
                        "Booking Value": [booking_value],
                        "Vehicle Type": [vehicle_type],
                        "Hour": [hour]
                    })

                    for col, le in label_encoders.items():
                        if col in input_df.columns:
                            input_df[col] = le.transform(input_df[col].astype(str))

                    input_scaled = scaler.transform(input_df)
                    pred = model.predict(input_scaled)
                    pred_label = target_encoder.inverse_transform(pred)
                    st.success(f"🎯 Predicted Booking Status: **{pred_label[0]}**")

            else:
                st.warning("⚠️ No rows left after filtering. Please adjust filters to see predictions.")
    else:
        st.info("👆 Upload a CSV file to get started.")
