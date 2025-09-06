"""
Streamlit Application for Ride Booking Completion Prediction.

This script creates a web interface using Streamlit to predict whether a ride
booking will be completed. It loads a pre-trained machine learning pipeline
(saved as 'booking_pipeline.joblib') that includes preprocessing steps and
the trained model. Users can input ride details through the interface,
and the application will use the pipeline to generate a prediction.
"""
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import time # Import time specifically for type hinting/clarity

# --- Pipeline Loading ---
# Load the saved machine learning pipeline which contains the preprocessor and the trained model.
try:
    pipeline = joblib.load('booking_pipeline.joblib')
    st.success("Machine learning pipeline loaded successfully!")
except FileNotFoundError:
    st.error("Error: 'booking_pipeline.joblib' not found. Please ensure the pipeline file is in the same directory.")
    pipeline = None # Set pipeline to None if loading fails to prevent further errors

# --- Streamlit App Layout and Input Fields ---
if pipeline: # Proceed only if the pipeline was loaded successfully
    # Set the title of the Streamlit application
    st.title('Ride Booking Completion Prediction')
    # Provide a brief description of the application's purpose
    st.write('Enter the details of the ride booking to predict if it will be completed.')

    st.subheader("Ride Details")

    # Input field for the Date of the booking
    date_input = st.date_input("Date")
    # Input field for the Time of the booking
    time_input = st.time_input("Time")

    # Combine date and time inputs to extract engineered features: weekday, is_weekend, and hour.
    if date_input and time_input:
        # Combine date and time into a single datetime object
        datetime_combined = pd.to_datetime(str(date_input) + ' ' + str(time_input))
        # Extract the day of the week (0=Monday, 6=Sunday)
        weekday = datetime_combined.dayofweek
        # Determine if it's a weekend (1 if Saturday or Sunday, 0 otherwise)
        is_weekend = int(weekday >= 5)
        # Extract the hour of the day
        hour = datetime_combined.hour
        # Optionally display the calculated features for user verification
        # st.write(f"Calculated features: Weekday={weekday}, Is_Weekend={is_weekend}, Hour={hour})
    else:
        # Initialize engineered features to None if date or time is not selected
        weekday, is_weekend, hour = None, None, None
        st.warning("Please select both Date and Time.") # Prompt user to select date and time

    # Input fields for original numeric features that were used in training.
    # Set appropriate min/max values and default values based on EDA.
    avg_vta = st.number_input("Avg VTAT", min_value=0.0, max_value=20.0, value=8.3, help="Average Vehicle Arrival Time")
    avg_cta = st.number_input("Avg CTAT", min_value=10.0, max_value=45.0, value=28.8, help="Average Customer Arrival Time")
    booking_value = st.number_input("Booking Value", min_value=50.0, max_value=4500.0, value=508.0, help="Monetary value of the booking")
    ride_distance = st.number_input("Ride Distance (km)", min_value=1.0, max_value=50.0, value=24.0, help="Distance of the ride in kilometers")
    driver_ratings = st.number_input("Driver Ratings (3.0-5.0)", min_value=3.0, max_value=5.0, value=4.2, step=0.1, help="Average rating of the driver")
    customer_rating = st.number_input("Customer Rating (3.0-5.0)", min_value=3.0, max_value=5.0, value=4.4, step=0.1, help="Average rating given by the customer")

    # Calculate the 'value_per_km' feature based on user inputs.
    # Handle division by zero or missing values (though number_input has default value).
    value_per_km = booking_value / ride_distance if ride_distance > 0 else 0
    # Optionally display the calculated value_per_km
    # st.write(f"Calculated value per km: {value_per_km:.2f})

    # Input fields for original categorical features using selectbox.
    # Provide options based on the unique values found in the training data.
    vehicle_type_options = ['Auto', 'Go Mini', 'Go Sedan', 'Bike', 'Premier Sedan', 'eBike', 'Uber XL']
    vehicle_type = st.selectbox("Vehicle Type", vehicle_type_options)

    # For locations, use a subset of common options or include 'missing'.
    # In a production app, this list might be loaded dynamically or be much larger with search functionality.
    location_options = ['Palam Vihar', 'Shastri Nagar', 'Khandsa', 'Central Secretariat', 'Ghitorni Village',
                        'Jhilmil', 'Gurgaon Sector 56', 'Malviya Nagar', 'Inderlok', 'Khan Market', 'missing'] # Include 'missing' placeholder
    pickup_location = st.selectbox("Pickup Location", location_options)
    drop_location = st.selectbox("Drop Location", location_options)

    # Options for payment method, including 'missing' placeholder.
    payment_method_options = ['UPI', 'Cash', 'Uber Wallet', 'Credit Card', 'Debit Card', 'missing']
    payment_method = st.selectbox("Payment Method", payment_method_options)

    # Note: Features like 'Cancelled Rides by Customer', 'Reason for cancelling by Customer',
    # 'Cancelled Rides by Driver', 'Driver Cancellation Reason', 'Incomplete Rides',
    # and 'Incomplete Rides Reason' were treated as potentially leakage-prone or
    # had high missing values and were handled by the preprocessor with default/missing values.
    # They are not taken as direct user inputs in this app to avoid complexity and potential leakage.
    # We will add placeholder values for these columns when creating the input DataFrame for prediction.

    # --- Prediction Button and Logic ---
    # Create a button to trigger the prediction
    if st.button('Predict Booking Completion'):
        # Ensure date and time were selected before proceeding with prediction
        if weekday is not None:
            # Create a Pandas DataFrame from the user inputs and engineered features.
            # This DataFrame must have the same column names as the data (X)
            # that the pipeline's preprocessor was fitted on.
            # Include all columns that were present in X, filling non-input columns
            # with appropriate default values (e.g., np.nan for numeric, 'missing' for categorical).
            # The preprocessor within the pipeline will handle the imputation and encoding.

            # Manually list the columns expected by the pipeline based on the training data structure (X).
            # This list should match X.columns.tolist() from the notebook's feature selection step.
            # A more robust approach would be to save this list alongside the pipeline.
            # Based on the notebook code, X contained original numeric, engineered numeric,
            # original categorical, and the 'other' columns (cancelled/incomplete flags and reasons).
            expected_input_columns = [
                'Avg VTAT', 'Avg CTAT', 'Cancelled Rides by Customer',
                'Reason for cancelling by Customer', 'Cancelled Rides by Driver',
                'Driver Cancellation Reason', 'Incomplete Rides', 'Incomplete Rides Reason',
                'Booking Value', 'Ride Distance', 'Driver Ratings', 'Customer Rating',
                'Payment Method', 'Weekday', 'Is_Weekend', 'Hour', 'value_per_km',
                'Vehicle Type', 'Pickup Location', 'Drop Location'
            ]

            # Create an empty DataFrame with the expected columns
            input_df = pd.DataFrame(columns=expected_input_columns)

            # Populate the DataFrame with user inputs and default values
            input_df.loc[0, 'Avg VTAT'] = avg_vta
            input_df.loc[0, 'Avg CTAT'] = avg_cta
            input_df.loc[0, 'Booking Value'] = booking_value
            input_df.loc[0, 'Ride Distance'] = ride_distance
            input_df.loc[0, 'Driver Ratings'] = driver_ratings
            input_df.loc[0, 'Customer Rating'] = customer_rating
            input_df.loc[0, 'Vehicle Type'] = vehicle_type
            input_df.loc[0, 'Pickup Location'] = pickup_location
            input_df.loc[0, 'Drop Location'] = drop_location
            input_df.loc[0, 'Payment Method'] = payment_method
            input_df.loc[0, 'Weekday'] = weekday
            input_df.loc[0, 'Is_Weekend'] = is_weekend
            input_df.loc[0, 'Hour'] = hour
            input_df.loc[0, 'value_per_km'] = value_per_km

            # Fill in default/placeholder values for columns not taken as direct input
            input_df.loc[0, 'Cancelled Rides by Customer'] = np.nan
            input_df.loc[0, 'Reason for cancelling by Customer'] = 'missing'
            input_df.loc[0, 'Cancelled Rides by Driver'] = np.nan
            input_df.loc[0, 'Driver Cancellation Reason'] = 'missing'
            input_df.loc[0, 'Incomplete Rides'] = np.nan
            input_df.loc[0, 'Incomplete Rides Reason'] = 'missing'

            # Ensure correct data types for the input DataFrame columns before prediction.
            # This is important because the preprocessor expects specific types (e.g., float64 for numeric).
            # Convert numeric columns to float type, handling potential None values if date/time weren't selected (though checked above)
            numeric_cols_in_input = ['Avg VTAT', 'Avg CTAT', 'Cancelled Rides by Customer',
                                     'Cancelled Rides by Driver', 'Incomplete Rides',
                                     'Booking Value', 'Ride Distance', 'Driver Ratings',
                                     'Customer Rating', 'Weekday', 'Is_Weekend', 'Hour', 'value_per_km']
            for col in numeric_cols_in_input:
                 if col in input_df.columns:
                    # Convert to float, coercing errors to NaN if necessary (though imputation handles NaN)
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')


            # Make prediction using the loaded pipeline.
            # The pipeline's preprocessor will transform the input_df, and the classifier will make the prediction.
            prediction = pipeline.predict(input_df)
            # Get the probability of the positive class (Completed, which is class 1)
            prediction_proba = pipeline.predict_proba(input_df)[:, 1]

            # --- Display Prediction Result ---
            st.subheader("Prediction Result")
            # Check the prediction result and display the outcome
            if prediction[0] == 1:
                st.success(f'Prediction: Booking is likely to be Completed')
                st.info(f'Probability of Completion: {prediction_proba[0]:.2f}')
            else:
                st.info(f'Prediction: Booking is likely Not Completed')
                st.info(f'Probability of Completion: {prediction_proba[0]:.2f}')

        else:
             st.warning("Please select both Date and Time before predicting.") # Reminder if date/time is missing

# --- Sidebar Instructions ---
# Add instructions on how to run the app in the sidebar
st.sidebar.subheader("How to Run")
st.sidebar.write("1. Save the code as `app.py`.")
st.sidebar.write("2. Ensure `booking_pipeline.joblib` is in the same directory.")
st.sidebar.write("3. Run `streamlit run app.py` in your terminal.")
