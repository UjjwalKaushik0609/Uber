import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

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
        A cleaned Pandas DataFrame, or None if an error occurs.
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

    df.drop_duplicates(inplace=True)
    df.fillna("Unknown", inplace=True)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    leakage_cols = [
        "Reason for cancelling by Customer", "Driver Cancellation Reason",
        "Cancelled Rides by Customer", "Cancelled Rides by Driver",
        "Incomplete Rides", "Incomplete Rides Reason"
    ]
    df.drop(columns=[col for col in leakage_cols if col in df.columns],
            inplace=True, errors="ignore")

    numeric_cols = ["Driver Ratings", "Customer Rating", "Booking Value", "Ride Distance", "Avg VTAT", "Avg CTAT"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if not df[col].empty and not df[col].isna().all():
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(0.0, inplace=True)
    
    st.success("✅ Data cleaned and preprocessed successfully!")
    return df

# --- Main application logic ---
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file:
    df = load_and_clean_data(uploaded_file)
    
    if df is not None:
        st.header("1. Data Overview")
        st.info("A preview of the raw data after initial cleaning.")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv,
            file_name="cleaned_ride_bookings.csv",
            mime="text/csv",
        )
        
        st.header("2. Exploratory Data Analysis (EDA)")

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

        if "Booking Status" in df.columns:
            with col1:
                st.subheader("Booking Status Distribution")
                fig, ax = plt.subplots()
                sns.countplot(x="Booking Status", data=df, ax=ax, palette="viridis")
                ax.set_title("Booking Status")
                st.pyplot(fig)

            st.markdown("---")

        st.subheader("Ratings Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if "Driver Ratings" in df.columns:
            ratings_numeric = df["Driver Ratings"].dropna()
            if not ratings_numeric.empty:
                sns.histplot(ratings_numeric, kde=True, ax=axes[0], color="blue")
                axes[0].set_title("Driver Ratings")
            else:
                axes[0].set_title("Driver Ratings (No Data)")
                axes[0].text(0.5, 0.5, 'No numeric data to display', ha='center', va='center')

        if "Customer Rating" in df.columns:
            ratings_numeric = df["Customer Rating"].dropna()
            if not ratings_numeric.empty:
                sns.histplot(ratings_numeric, kde=True, ax=axes[1], color="green")
                axes[1].set_title("Customer Ratings")
            else:
                axes[1].set_title("Customer Rating (No Data)")
                axes[1].text(0.5, 0.5, 'No numeric data to display', ha='center', va='center')
        
        st.pyplot(fig)
        st.markdown("---")

        # --- Section 3: Machine Learning Prediction ---
        st.header("3. Machine Learning Prediction")
        st.info("Predict the status of a new booking based on its features.")
        
        target_col = "Booking Status"
        
        if target_col in df.columns:
            X = df.drop(columns=[target_col, "Date", "Time", "Booking ID", "Customer ID"], errors='ignore')
            y = df[target_col]

            @st.cache_data
            def preprocess_for_ml(data, target_y):
                le_dict = {}
                X_encoded = data.copy()
                
                for col in X_encoded.columns:
                    if X_encoded[col].dtype == "object":
                        le = LabelEncoder()
                        X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                        le_dict[col] = le
                
                # FIX: Explicitly encode the target variable and store its encoder
                le_target = LabelEncoder()
                y_encoded = le_target.fit_transform(target_y.astype(str))
                le_dict[target_y.name] = le_target
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_encoded)
                
                unique_classes, counts = np.unique(y_encoded, return_counts=True)
                scale_pos_weight = 1
                if len(counts) == 2:
                    pos_class_index = np.argmin(counts)
                    neg_class_index = np.argmax(counts)
                    scale_pos_weight = counts[neg_class_index] / counts[pos_class_index]
                
                return X_scaled, le_dict, scaler, scale_pos_weight, y_encoded

            X_scaled, label_encoders, scaler, scale_pos_weight, y_encoded = preprocess_for_ml(X, y)
            X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )

            models = {
                "XGBoost Classifier": XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, use_label_encoder=False, eval_metric='logloss'),
            }

            st.subheader("Model Performance")
            st.markdown("⚠️ **Note:** The XGBoost model is trained to specifically handle imbalanced data, improving predictions for the minority class ('Incomplete').")
            model_choice = st.selectbox("Choose a model to train:", list(models.keys()))
            model = models[model_choice]

            model.fit(X_train, y_train_encoded)
            preds_encoded = model.predict(X_test)
            
            preds = label_encoders[target_col].inverse_transform(preds_encoded)
            y_test_original = label_encoders[target_col].inverse_transform(y_test_encoded)
            
            acc = accuracy_score(y_test_original, preds) * 100

            st.markdown(f"**Selected Model:** `{model_choice}`")
            st.success(f"**Accuracy:** `{acc:.2f}%`")

            st.subheader("Detailed Performance Metrics")
            report = classification_report(y_test_original, preds, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='YlOrRd'))
            
            st.markdown("---")

            st.subheader("Predict a New Booking")
            st.markdown("Enter the details for a new ride to get a prediction.")

            input_data = {}
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Pickup Location' in df.columns:
                    input_data['Pickup Location'] = st.selectbox(
                        "Select Pickup Location",
                        options=[''] + sorted(list(df['Pickup Location'].unique())),
                        help="Choose the pickup point for the new ride."
                    )
                if 'Drop Location' in df.columns:
                    input_data['Drop Location'] = st.selectbox(
                        "Select Drop Location",
                        options=[''] + sorted(list(df['Drop Location'].unique())),
                        help="Choose the destination for the new ride."
                    )
                if 'Vehicle Type' in df.columns:
                    input_data['Vehicle Type'] = st.selectbox(
                        "Select Vehicle Type",
                        options=sorted(list(df['Vehicle Type'].unique())),
                        help="Choose the type of vehicle (e.g., Bike, Auto, Car)."
                    )

            with col2:
                if 'Ride Distance' in df.columns:
                    input_data['Ride Distance'] = st.number_input(
                        "Ride Distance (km)",
                        min_value=0.1,
                        max_value=100.0,
                        value=df['Ride Distance'].mean() if not df['Ride Distance'].empty and not df['Ride Distance'].isna().all() else 5.0,
                        help="Enter the estimated distance of the ride."
                    )
                if 'Customer Rating' in df.columns:
                    input_data['Customer Rating'] = st.slider(
                        "Customer Rating",
                        min_value=1.0,
                        max_value=5.0,
                        value=df['Customer Rating'].mean() if not df['Customer Rating'].empty and not df['Customer Rating'].isna().all() else 4.5,
                        help="Enter the customer's rating (1-5)."
                    )
                if 'Payment Method' in df.columns:
                    input_data['Payment Method'] = st.selectbox(
                        "Select Payment Method",
                        options=sorted(list(df['Payment Method'].unique())),
                        help="Choose the payment method for the booking."
                    )

            if st.button("✨ Predict Booking Status"):
                if 'Pickup Location' in input_data and not input_data['Pickup Location']:
                    st.warning("Please select a Pickup Location.")
                elif 'Drop Location' in input_data and not input_data['Drop Location']:
                    st.warning("Please select a Drop Location.")
                else:
                    final_input_df = pd.DataFrame(columns=X.columns)
                    final_input_df.loc[0] = 0
                    
                    for key, value in input_data.items():
                        if key in final_input_df.columns:
                            final_input_df.loc[0, key] = value
                    
                    for col in X.columns:
                        if col not in final_input_df.columns:
                            final_input_df[col] = df[col].mean() if df[col].dtype != 'object' else 'Unknown'

                    for col in final_input_df.columns:
                        if final_input_df[col].dtype == 'object' and col in label_encoders:
                            try:
                                final_input_df[col] = label_encoders[col].transform(final_input_df[col])
                            except ValueError as e:
                                st.error(f"Cannot encode value for column '{col}': {e}. Please ensure selected value is present in the training data.")
                                st.stop()
                    
                    input_scaled = scaler.transform(final_input_df)
                    prediction_encoded = model.predict(input_scaled)
                    
                    prediction = label_encoders[target_col].inverse_transform(prediction_encoded)
                    
                    st.success(f"**Predicted Booking Status:** `{prediction[0]}` 🎉")

else:
    st.info("👆 Upload a CSV file to get started.")
