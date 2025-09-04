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

# Streamlit Configuration
st.set_page_config(
    page_title="Uber Ride Analytics & Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚖 Uber Ride Analytics & Booking Prediction")
st.markdown(
    """
    Explore and analyze your Uber ride data, then predict the booking status using a
    range of machine learning models.
    """
)

# Function to load and preprocess data
@st.cache_data
def load_data(uploaded_file):
    """Loads and cleans the uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    df.drop_duplicates(inplace=True)
    df.fillna("Unknown", inplace=True)

    # Convert date and time columns
    for col in ["Date", "Time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

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
    return df

# Main application logic
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)

    st.sidebar.success("✅ Data loaded successfully!")

    # Display raw data
    st.header("1. Data Overview")
    st.info("A preview of the raw data from your uploaded file.")
    st.dataframe(df.head())

    # --- Section for EDA ---
    st.header("2. Exploratory Data Analysis (EDA)")
    
    # Daily rides and revenue
    if "Date" in df.columns and "Booking Value" in df.columns:
        st.subheader("Daily Rides & Revenue Over Time")
        df_eda = df.copy()
        df_eda["Booking Value"] = pd.to_numeric(df_eda["Booking Value"], errors="coerce").fillna(0)
        daily_stats = (
            df_eda.groupby(df_eda["Date"].dt.date)
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
            sns.countplot(
                x="Booking Status", data=df, ax=ax, palette="viridis"
            )
            ax.set_title("Booking Status")
            st.pyplot(fig)

    # Vehicle Type Distribution
    if "Vehicle Type" in df.columns:
        with col2:
            st.subheader("Vehicle Type Popularity")
            fig, ax = plt.subplots()
            sns.countplot(
                y="Vehicle Type", data=df, order=df['Vehicle Type'].value_counts().index, palette="coolwarm"
            )
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
        
    # Ratings
    col_ratings1, col_ratings2 = st.columns(2)
    if "Driver Ratings" in df.columns:
        with col_ratings1:
            st.subheader("Driver Ratings Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["Driver Ratings"], kde=True, ax=ax, color="blue")
            st.pyplot(fig)
            
    if "Customer Rating" in df.columns:
        with col_ratings2:
            st.subheader("Customer Ratings Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["Customer Rating"], kde=True, ax=ax, color="green")
            st.pyplot(fig)
    st.markdown("---")

    # --- Section for ML Prediction ---
    st.header("3. Machine Learning Prediction")
    st.info("Predict the status of a new booking based on its features.")
    
    target_col = "Booking Status"
    
    if target_col in df.columns:
        # Pre-process data for ML
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Use st.cache_data to speed up preprocessing
        @st.cache_data
        def preprocess_for_ml(data, target_y):
            le_dict = {}
            X_encoded = data.copy()
            for col in X_encoded.columns:
                if X_encoded[col].dtype == "object":
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    le_dict[col] = le
            
            # Drop date-time columns before scaling
            X_encoded = X_encoded.select_dtypes(include=[np.number])
            
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
        model_choice = st.selectbox("Choose a model:", list(models.keys()))
        model = models[model_choice]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds) * 100

        st.markdown(f"**Selected Model:** {model_choice}")
        st.success(f"**Accuracy:** {acc:.2f}%")
        
        # Display feature importance for tree-based models
        if model_choice in ["Decision Tree", "Random Forest"]:
            st.markdown("---")
            st.subheader("Feature Importance")
            feature_importances = pd.Series(
                model.feature_importances_, index=df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importances.sort_values(ascending=False).head(10).plot(
                kind="barh", ax=ax
            )
            ax.set_title("Top 10 Most Important Features")
            st.pyplot(fig)
            st.markdown("---")
        
        # New prediction section for user input
        st.subheader("Predict a New Booking")
        st.markdown("Enter the details of a new ride to get a prediction.")

        # Create input fields based on available columns
        input_data = {}
        numeric_cols = df.select_dtypes(include=np.number).columns.drop([target_col, "Year", "Month", "Day", "Hour", "Minute"], errors='ignore')
        object_cols = df.select_dtypes(include="object").columns.drop([target_col], errors='ignore')
        
        if "Pickup Location" in object_cols:
            pickup_loc = st.selectbox("Pickup Location", [""] + sorted(df["Pickup Location"].unique().tolist()))
            input_data["Pickup Location"] = pickup_loc
        
        if "Drop Location" in object_cols:
            drop_loc = st.selectbox("Drop Location", [""] + sorted(df["Drop Location"].unique().tolist()))
            input_data["Drop Location"] = drop_loc
            
        for col in numeric_cols:
            input_data[col] = st.number_input(f"Enter {col}", value=df[col].mean())
            
        for col in object_cols:
            if col not in ["Pickup Location", "Drop Location"]:
                input_data[col] = st.selectbox(f"Select {col}", [""] + sorted(df[col].unique().tolist()))

        if st.button("Predict Booking Status"):
            if not all(val for val in input_data.values()):
                st.warning("Please fill in all the required fields.")
            else:
                input_df = pd.DataFrame([input_data])
                
                # Align columns and encode new data
                for col in X.columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                input_df = input_df[X.columns]
                
                for col in input_df.columns:
                    if input_df[col].dtype == "object" and col in label_encoders:
                        try:
                            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                        except ValueError:
                            st.error(f"Cannot encode '{input_df[col].iloc[0]}' for column '{col}'. It's not in the training data.")
                            st.stop()
                
                # Scale the input data
                input_df_scaled = scaler.transform(input_df.select_dtypes(include=[np.number]))
                
                # Predict
                prediction = model.predict(input_df_scaled)
                prediction_label = prediction[0]
                
                st.success(f"**Predicted Booking Status:** {prediction_label} 🎉")


else:
    st.info("👆 Upload a CSV file to get started with the analysis and prediction.")
