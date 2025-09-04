🚕 NCR Ride Bookings – Data Analysis & Prediction

🔗 Live App: Click here to open 🚀

📌 Objective

Analyze NCR ride booking data to explore ride patterns, cancellations, and revenues.
Build ML models to predict booking status and deploy results as an interactive Streamlit app.

📂 Dataset Overview

File: ncr_ride_bookings.csv

Records: Several thousand

Columns: Ride details, locations, vehicle types, values, ratings, cancellations

📊 Data Dictionary
✅ Columns Used for Training
Column Name	Type	Description
Year	int	Extracted from booking date
Month	int	Extracted from booking date
Day	int	Extracted from booking date
Hour	int	Extracted from booking time
Minute	int	Extracted from booking time
Booking Value	float	Fare amount (numeric)
Vehicle Type	category	Cab / Auto / Bike, etc.
Pickup Location	category	Starting point
Drop Location	category	Destination point
Ride Distance	float	Distance travelled
Avg VTAT	float	Avg. Vehicle Time to Arrival
Avg CTAT	float	Avg. Customer Time to Arrival

🎯 Target Column:

Booking Status (Completed / Cancelled / Incomplete)

❌ Columns Dropped (Reasons)
Column Name	Reason
Booking ID	Unique identifier
Date	Replaced with Year, Month, Day
Time	Replaced with Hour, Minute
Reason for cancelling by Customer	Leakage (reveals outcome)
Driver Cancellation Reason	Leakage
Customer Rating	Leakage (given after ride)
Driver Ratings	Leakage
Cancelled Rides by Customer	Leakage
Cancelled Rides by Driver	Leakage
Incomplete Rides	Leakage
Incomplete Rides Reason	Leakage
🧹 Data Cleaning

Removed duplicates

Handled missing values (Unknown / 0)

Converted Booking Value → numeric

Extracted Year, Month, Day, Hour, Minute

Dropped leakage columns

📈 Exploratory Data Analysis

The app provides interactive dashboards for:

📊 Daily Rides & Revenue

✅ Booking Status Distribution

🚗 Vehicle Type Usage

📍 Top Pickup & Drop Locations

❌ Cancellation Reasons

⭐ Driver & Customer Ratings

🔥 Correlation Heatmap

🤖 Machine Learning

Target: Booking Status

Models Tested:

Logistic Regression

Decision Tree

Random Forest

Naive Bayes

SVM

KNN

(Optional) XGBoost

Results:

🌟 Random Forest & XGBoost → highest accuracy (~80–90%)

Logistic Regression → solid but slightly lower

Naive Bayes → weaker due to categorical imbalance

🌐 Streamlit App Features

📂 Upload raw CSV

🧹 Auto-clean data

🔎 Interactive filters (date, vehicle type, booking status)

📥 Download cleaned dataset

📊 EDA dashboards

🤖 Choose ML model → see accuracy

📌 Feature importance (for tree-based models)

🚀 Deployment
Run Locally
pip install -r requirements.txt
streamlit run app.py

Deploy to Streamlit Cloud

Push app.py, requirements.txt (and optional dataset) to GitHub

Go to Streamlit Cloud

Create new app → connect GitHub → choose app.py

Deploy 🚀

✅ Live App: Click here

🔑 Key Insights

❌ Cancellations cluster around certain pickup points

💰 Revenue strongly linked to ride distance

🛵 Autos & bikes dominate short urban trips

🔮 Prediction helps forecast booking outcomes

📌 Future Work

Real-time API integration

Geospatial analysis with maps

Deep learning models (LSTM) for time-series forecasting

Customer churn analysis
