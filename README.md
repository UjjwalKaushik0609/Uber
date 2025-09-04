🚕 NCR Ride Bookings – Data Analysis & Prediction Project

📍 Domain: Ride-hailing / Mobility
🔗 Live App: NCR Ride Prediction Dashboard

This project simulates a real-world data science and analytics case study using NCR ride booking data.
It covers the full pipeline — from data cleaning & visualization to machine learning prediction models, and finally deployment as a Streamlit web app.

🚀 Who This Project Is For

📊 Data Analyst aspirants wanting an end-to-end project for their portfolio

🤖 Data Science learners looking to apply ML models to categorical + numerical data

💼 Candidates preparing for analytics/DS interviews in mobility, logistics, or e-commerce domains

📦 Dataset Overview

The dataset (ncr_ride_bookings.csv) contains several thousand ride bookings in NCR, with details on vehicles, routes, values, ratings, and cancellations.

🧾 Columns Used for Training
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

❌ Columns Dropped
Column Name	Reason
Booking ID	Unique identifier
Date, Time	Replaced by derived features
Reason for cancelling by Customer	Leakage (reveals outcome)
Driver Cancellation Reason	Leakage
Customer Rating, Driver Ratings	Given post-ride (leakage)
Cancelled Rides by Customer	Leakage
Cancelled Rides by Driver	Leakage
Incomplete Rides, Incomplete Rides Reason	Leakage
🔧 Project Workflow

1️⃣ Data Cleaning

Removed duplicates and handled missing values (Unknown/0)

Converted Booking Value → numeric

Extracted Year, Month, Day, Hour, Minute

Dropped leakage columns

2️⃣ Exploratory Data Analysis (EDA)

📊 Daily rides & revenue over time

✅ Booking status distribution

🚗 Vehicle type usage patterns

📍 Top 10 pickup & drop locations

❌ Cancellation reasons (customer vs driver)

⭐ Ratings distribution (customer & driver)

🔥 Correlation heatmap of numeric features

3️⃣ Machine Learning Models

Logistic Regression

Decision Tree

Random Forest

Naive Bayes

SVM

KNN

(Optional) XGBoost

4️⃣ Deployment

Interactive dashboards in Streamlit

Model performance comparison charts

CSV upload → auto-cleaning → EDA + ML results

📊 Model Results
Model	Accuracy (%)
Logistic Regression	~78%
Decision Tree	~82%
Random Forest	~89%
Naive Bayes	~70%
SVM	~80%
KNN	~76%
XGBoost (if enabled)	~90%

👉 Random Forest & XGBoost consistently gave the best performance.

🖥️ Streamlit App Features

📂 Upload your own NCR ride booking CSV

🧹 Auto data cleaning & preprocessing

📊 Interactive dashboards for EDA

🔎 Explore cancellations, vehicle usage, revenue trends

🤖 Train/test ML models and view accuracy results

📥 Download cleaned dataset

🔗 Try it Live: NCR Ride Prediction Dashboard

🔑 Key Insights

❌ Cancellations cluster around specific pickup points

💰 Revenue is strongly tied to ride distance

🛵 Autos & bikes dominate short intra-city rides

🔮 Predictive models can forecast booking outcomes with ~90% accuracy

🧠 Key Learnings

Handling categorical + numeric preprocessing in scikit-learn

Identifying and removing data leakage columns

Building an ML pipeline with multiple models for comparison

Deploying ML + EDA together in a Streamlit dashboard

🧰 Tools Used

🐼 pandas, numpy

📊 seaborn, matplotlib

🤖 scikit-learn, (optional) xgboost

🌐 Streamlit for app & deployment

💻 GitHub for version control

🚀 Deployment Guide
Run Locally
pip install -r requirements.txt
streamlit run app.py

Deploy on Streamlit Cloud

Push app.py + requirements.txt to GitHub

Go to Streamlit Cloud

Create a new app → connect your repo → select app.py

Deploy 🚀

Live link: https://2aqyfhdehiughnvkzcvgb4.streamlit.app/

✨ End-to-end workflow: data → cleaning → EDA → ML → Streamlit deployment.
