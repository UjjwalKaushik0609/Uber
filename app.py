# app.py
import streamlit as st
import pandas as pd
import numpy as np
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
    df.drop(columns=[col for col in leakage_cols if col in df.columns], inplace=True, errors="ignore")

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
            df = df[(df["Date"] >= pd.to_datetime(date_range[0])) & (df["Date"] <= pd.to_datetime(date_range[1]))]

    if "Vehicle Type" in df.columns:
        vehicle_filter = st.sidebar.multiselect("Select Vehicle Types", df["Vehicle Type"].unique(), default=df["Vehicle Type"].unique())
        df = df[df["Vehicle Type"].isin(vehicle_filter)]

    if "Booking Status" in df.columns:
        status_filter = st.sidebar.multiselect("Select Booking Status", df["Booking Status"].unique(), default=df["Booking Status"].unique())
        df = df[df["Booking Status"].isin(status_filter)]

    # -------------------------------
    # EDA Visualizations (Matplotlib only)
    # -------------------------------
    st.subheader("📈 Exploratory Data Analysis")

    # 1️⃣ Daily rides & revenue
    if "Date" in df.columns and "Booking Value" in df.columns:
        st.write("### Daily Rides & Revenue")
        df["Booking Value"] = pd.to_numeric(df["Booking Value"], errors="coerce").fillna(0)
        rides_per_day = df.groupby("Date").size().reset_index(name="Total Rides")
        revenue_per_day = df.groupby("Date")["Booking Value"].sum().reset_index(name="Total Revenue")
        daily_stats = pd.merge(rides_per_day, revenue_per_day, on="Date")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(daily_stats["Date"], daily_stats["Total Rides"], label="Total Rides", color="blue")
        ax.plot(daily_stats["Date"], daily_stats["Total Revenue"], label="Total Revenue", color="green")
        ax.set_title("Daily Rides & Revenue")
        ax.legend()
        st.pyplot(fig)

    # 2️⃣ Booking Status
    if "Booking Status" in df.columns:
        st.write("### Booking Status Distribution")
        fig, ax = plt.subplots()
        counts = df["Booking Status"].value_counts()
        ax.bar(counts.index, counts.values, color="skyblue")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # 3️⃣ Vehicle Type
    if "Vehicle Type" in df.columns:
        st.write("### Vehicle Type Distribution")
        fig, ax = plt.subplots()
        counts = df["Vehicle Type"].value_counts()
        ax.bar(counts.index, counts.values, color="orange")
        plt.xticks(rotation=45)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # 4️⃣ Top Pickup & Drop
    if "Pickup Location" in df.columns and "Drop Location" in df.columns:
        st.write("### Top 10 Pickup & Drop Locations")
        top_pickups = df["Pickup Location"].value_counts().head(10)
        top_drops = df["Drop Location"].value_counts().head(10)
        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        axes[0].barh(top_pickups.index[::-1], top_pickups.values[::-1], color="blue")
        axes[0].set_title("Top Pickup Locations")
        axes[1].barh(top_drops.index[::-1], top_drops.values[::-1], color="green")
        axes[1].set_title("Top Drop Locations")
        st.pyplot(fig)

    # 5️⃣ Ratings
    if "Driver Ratings" in df.columns or "Customer Rating" in df.columns:
        st.write("### Ratings Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12,5))
        if "Driver Ratings" in df.columns:
            axes[0].hist(df["Driver Ratings"].dropna(), bins=10, color="blue", alpha=0.7)
            axes[0].set_title("Driver Ratings")
        if "Customer Rating" in df.columns:
            axes[1].hist(df["Customer Rating"].dropna(), bins=10, color="green", alpha=0.7)
            axes[1].set_title("Customer Ratings")
        st.pyplot(fig)

    # 6️⃣ Correlation Heatmap (matplotlib imshow)
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 1:
        st.write("### Correlation Heatmap")
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        cax = ax.matshow(corr, cmap="coolwarm")
        fig.colorbar(cax)
        ax.set_xticks(range(len(num_cols)))
        ax.set_yticks(range(len(num_cols)))
        ax.set_xticklabels(num_cols, rotation=90)
        ax.set_yticklabels(num_cols)
        st.pyplot(fig)

    # -------------------------------
    # ML Prediction (All Models)
    # -------------------------------
    st.subheader("🤖 ML Model Performance on Booking Status")

    target_col = "Booking Status"
    if target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier()
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds) * 100
            results[name] = acc
            st.success(f"✅ {name} Accuracy: {acc:.2f}%")

            if name in ["Decision Tree", "Random Forest"]:
                st.write(f"### {name} Feature Importance")
                feature_importances = pd.Series(model.feature_importances_, index=df.drop(columns=[target_col]).columns)
                fig, ax = plt.subplots()
                feature_importances.sort_values(ascending=False).head(10).plot(kind="bar", ax=ax, color="purple")
                st.pyplot(fig)

        # Compare all models in one bar chart
        st.write("### 📊 Model Accuracy Comparison")
        fig, ax = plt.subplots()
        ax.bar(results.keys(), results.values(), color="teal")
        ax.set_ylabel("Accuracy (%)")
        plt.xticks(rotation=45)
        st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to get started.")
