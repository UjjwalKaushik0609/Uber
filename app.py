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

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Uber Ride Analytics & Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚖 Uber Ride Analytics & Booking Prediction")
st.markdown(
    """
    Explore and analyze your Uber ride data, then predict the booking status using a
    range of machine learning models. Upload your CSV file to get started!
    """
)

# --- Data Loading and Preprocessing Function ---
@st.cache_data
def load_and_clean_data(uploaded_file):
    """
    Loads, cleans, and preprocesses the uploaded CSV file.

    Args:
        uploaded_file: A file-like object uploaded via Streamlit.

    Returns:
        A cleaned Pandas DataFrame.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

    # Drop duplicates and fill missing values
    df.drop_duplicates(inplace=True)
    df.fillna("Unknown", inplace=True)

    # Convert date and time columns
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    # Drop potential data leakage columns
    leakage_cols = [
        "Reason for cancelling by Customer",
        "Driver Cancellation Reason",
        "Cancelled Rides by Customer",
        "Cancelled Rides by Driver",
        "Incomplete Rides",
        "Incomplete Rides Reason",
    ]
    df.drop(
        columns=[col for col in leakage_cols if col in df.columns],
        inplace=True,
        errors="ignore",
    )

    # Convert ratings and booking value to numeric, handling errors
    for col in ["Driver Ratings", "Customer Rating", "Booking Value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(0, inplace=True)  # Fill NaNs created by coerce with 0 or mean

    st.success("✅ Data cleaned successfully!")
    return df

# --- Main application logic ---
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        st.sidebar.success("✅ Data loaded and cleaned!")

        # --- Section 1: Data Overview ---
        st.header("1. Data Overview")
        st.info("A preview of the raw data after initial cleaning.")
        st.dataframe(df.head())

        # Download cleaned data
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv,
            file_name="cleaned_ride_bookings.csv",
            mime="text/csv",
        )
        
        # --- Section 2: Exploratory Data Analysis (EDA) ---
        st.header("2. Exploratory Data Analysis (EDA)")

        # Daily rides and revenue
        if "Date" in df.columns and "Booking Value" in df.columns:
            st.subheader("Daily Rides & Revenue Over Time")
            
            daily_stats = (
                df.groupby(df["Date"].dt.date)
                .agg(Total_Rides=("Date", "size"), Total_Revenue=("Booking Value", "sum"))
                .reset_index()
            )
            st.line_chart(daily_stats.set_index("Date"))
        
        st.markdown("---")

        col1, col2 = st.columns(2)

        # Booking Status Distribution
        if "Booking Status" in df.columns:
            with col1:
                st.subheader("Booking Status Distribution")
                fig, ax = plt.subplots()
                sns.countplot(x="Booking Status", data=df, ax=ax, palette="viridis")
                ax.set_title("Booking Status")
                st.pyplot(fig)

        # Vehicle Type Distribution
        if "Vehicle Type" in df.columns:
            with col2:
                st.subheader("Vehicle Type Popularity")
                fig, ax = plt.subplots()
                sns.countplot(y="Vehicle Type", data=df, order=df['Vehicle Type'].value_counts().index, palette="coolwarm")
                ax.set_title("Vehicle Type")
                st.pyplot(fig)
        
        st.markdown("---")
        
        # Top Pickup & Drop
        if "Pickup Location" in df.columns and "Drop Location" in df.columns:
            st.subheader("Top Locations")
            top_pickups = df["Pickup Location"].value_counts().head(10)
            top_drops = df["Drop Location"].value_counts().head(10)
            
            col_pickup, col_drop = st.columns(2)
            
            with col_pickup:
                st.markdown("#### Top 10 Pickup Locations")
                st.bar_chart(top_pickups)
                
            with col_drop:
                st.markdown("#### Top 10 Drop Locations")
                st.bar_chart(top_drops)
        
        st.markdown("---")
        
        # Ratings Distribution (FIXED)
        st.subheader("Ratings Distribution")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot Driver Ratings
        if "Driver Ratings" in df.columns:
            ratings_numeric = df["Driver Ratings"].dropna()
            if not ratings_numeric.empty:
                sns.histplot(ratings_numeric, kde=True, ax=axes[0], color="blue")
                axes[0].set_title("Driver Ratings")
            else:
                axes[0].set_title("Driver Ratings (No Data)")
                axes[0].text(0.5, 0.5, 'No numeric data to display', ha='center', va='center')

        # Plot Customer Ratings
        if "Customer Rating" in df.columns:
            ratings_numeric = df["Customer Rating"].dropna()
            if not ratings_numeric.empty:
                sns.histplot(ratings_numeric, kde=True, ax=axes[1], color="green")
                axes[1].set_title("Customer Ratings")
            else:
                axes[1].set_title("Customer Ratings (No Data)")
                axes[1].text(0.5, 0.5, 'No numeric data to display', ha='center', va='center')
        
        st.pyplot(fig)
        st.markdown("---")

        # --- Section 3: Machine Learning Prediction ---
        st.header("3. Machine Learning Prediction")
        st.info("Predict the status of a new booking based on its features.")
        
        target_col = "Booking Status"
        
        if target_col in df.columns:
            # Pre-process data for ML
            X = df.drop(columns=[target_col, "Date", "Time"], errors='ignore')
            y = df[target_col]

            # Use st.cache_data to speed up preprocessing
            @st.cache_data
            def preprocess_for_ml(data, target_y):
                le_dict = {}
                X_encoded = data.copy()
                
                # Encode categorical columns
                for col in X_encoded.columns:
                    if X_encoded[col].dtype == "object":
                        le = LabelEncoder()
                        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                        le_dict[col] = le
                
                # Scale numeric columns
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_encoded)
                return X_scaled, le_dict, scaler

            X_scaled, label_encoders, scaler = preprocess_for_ml(X, y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            models = {
                "Random Forest": RandomForestClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "SVM": SVC(random_state=42),
                "KNN": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
            }

            # Model selection and training
            st.subheader("Model Performance")
            model_choice = st.selectbox("Choose a model to train:", list(models.keys()))
            model = models[model_choice]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds) * 100

            st.markdown(f"**Selected Model:** `{model_choice}`")
            st.success(f"**Accuracy:** `{acc:.2f}%`")
            
            # Display feature importance for tree-based models
            if model_choice in ["Decision Tree", "Random Forest"]:
                st.markdown("---")
                st.subheader("Feature Importance")
                feature_importances = pd.Series(
                    model.feature_importances_, index=X.columns
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                feature_importances.sort_values(ascending=False).head(10).plot(
                    kind="barh", ax=ax
                )
                ax.set_title("Top 10 Most Important Features")
                st.pyplot(fig)
            
            st.markdown("---")

            # --- Predict a New Booking Section ---
            st.subheader("Predict a New Booking")
            st.markdown("Enter the details for a new ride to get a prediction.")

            input_data = {}
            for col in X.columns:
                if col in label_encoders:
                    options = sorted(list(label_encoders[col].classes_))
                    input_data[col] = st.selectbox(f"Select {col}", options=options)
                else:
                    input_data[col] = st.number_input(f"Enter {col}", value=0.0)

            if st.button("Predict Booking Status"):
                # Create a DataFrame from user input
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical inputs
                for col in input_df.columns:
                    if col in label_encoders:
                        try:
                            # Use the same LabelEncoder instance
                            input_df[col] = label_encoders[col].transform(input_df[col])
                        except ValueError:
                            st.error(f"Cannot encode '{input_df[col].iloc[0]}' for column '{col}'. It was not in the original training data.")
                            st.stop()
                
                # Align columns and handle missing numeric features
                input_df = input_df.reindex(columns=X.columns, fill_value=0)
                
                # Scale the input data
                input_scaled = scaler.transform(input_df)
                
                # Predict and display the result
                prediction = model.predict(input_scaled)
                st.success(f"**Predicted Booking Status:** `{prediction[0]}` 🎉")

else:
    st.info("👆 Upload a CSV file to get started.")
