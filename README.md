# 🚖 Uber Ride Analytics & Prediction Project  

📍 **Domain:** Ride-hailing / Mobility  
🔗 **Live App:** [NCR Ride Prediction Dashboard](https://2aqyfhdehiughnvkzcvgb4.streamlit.app/)  

This project is an **end-to-end data science case study** analyzing **NCR Uber ride bookings**. It integrates **business analytics, machine learning, and dashboard deployment** to uncover:  

- 📊 **Customer behavior patterns**  
- 🚗 **Vehicle type preferences**  
- 💰 **Revenue & booking trends**  
- ❌ **Cancellations & service reliability issues**  
- 🤖 **Predictive models for booking success**  

---

## 🧰 Tech Stack  

- **Data Handling:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Machine Learning:** scikit-learn (LogReg, Decision Tree, Random Forest, Naive Bayes, KNN, SVM), XGBoost  
- **Deployment:** Streamlit (dashboard & model hosting)  
- **Version Control:** GitHub  

---

## 📦 Dataset Overview  

**File:** `ncr_ride_bookings.csv`  
**Rows:** Several thousand bookings (2024 NCR data)  

### Features Used  
- **Temporal:** Year, Month, Day, Hour, Minute  
- **Ride Details:** Booking Value (fare), Vehicle Type, Ride Distance  
- **Operational Metrics:** Avg VTAT (Vehicle Time to Arrival), Avg CTAT (Customer Time to Arrival)  
- **Geographic:** Pickup Location, Drop-off Location  

🎯 **Target Variable:** Booking Status (`Completed / Cancelled / Incomplete`)  

### Dropped (Leakage) Columns  
- Booking ID (unique only)  
- Date, Time (derived into Year/Month/Day/Hour/Minute)  
- Customer/Driver cancellation reasons (direct leakage)  
- Post-ride ratings (given after ride completion)  

---

## 🔎 Exploratory Data Analysis (EDA)  

### 1️⃣ Correlation & Temporal Features  
- Correlation heatmap → **temporal features (Year, Month, Hour)** have **near-zero impact** on booking value.  
- No multicollinearity issues.  
- 🚦 Demand & revenue are influenced more by **external factors (distance, wait times, location)** than by time of day.  

### 2️⃣ Vehicle Type Preferences  
| Vehicle Type | Rides | Market Share |
|--------------|-------|--------------|
| Auto 🚕 | 37,000+ | **31%** |
| Go Mini 🚗 | 30,000+ | **25%** |
| Go Sedan 🚘 | 27,000+ | **23%** |
| Bike 🏍️ | 22,500+ | 19% |
| Premier Sedan 🚖 | 18,000+ | 15% |
| eBike 🚲 | 10,500+ | 9% |
| Uber XL 🚐 | 4,500+ | 4% |

**Insights:**  
- **Autos & Go Mini dominate** affordable transport.  
- **Sedans capture premium demand** (healthy segment share).  
- **Uber XL adoption low** → demand limited to groups/events.  

### 3️⃣ Geographic Distribution  
- **Top Pickups:** Khandsa, Barakhamba Road, Saket, Pragati Maidan  
- **Top Drop-offs:** Ashram, Cyber Hub, Narsinghpur  
- 📍 **Hub-and-spoke demand** → mix of **residential & business clusters**  
- Event-driven surges around **Pragati Maidan**.  

### 4️⃣ Revenue & Ride Trends  
- Revenue peaks: **220K+ units** during high-demand periods.  
- Stable baseline: **120K–140K units** → market maturity.  
- Ride volume & revenue show **direct correlation**.  
- 📈 Growth trajectory evident in late 2024.  

### 5️⃣ Booking Completion Analysis  
| Status | Bookings | % of Total |
|--------|----------|------------|
| ✅ Completed | 92,000+ | **72%** |
| ❌ Cancelled by Customer | 26,000+ | 20% |
| ❌ Cancelled by Driver | 11,000+ | 8% |
| 🚫 No Driver Found | 11,000+ | 8% |
| 🕒 Incomplete | 9,000+ | 7% |

**Insights:**  
- **72% completion rate** = strong reliability.  
- **20% customer cancellations** → pricing/UX gaps.  
- **16% driver-side issues** (cancellations + supply shortages).  

---

## 🤖 Machine Learning Models  

### Model Performance  
| Model                | Accuracy (%) |
|-----------------------|-------------|
| Logistic Regression   | 87.3 |
| Decision Tree         | 89.0 |
| Random Forest         | **92.5** |
| Naive Bayes           | 87.4 |
| KNN                   | 80.6 |
| SVM                   | Dropped (low acc.) |
| XGBoost (optional)    | ~90.0 |

➡️ **Random Forest & XGBoost are the best predictors**  

### Feature Importance  
- **Random Forest:**  
  - Avg VTAT → 25%  
  - Ride Distance → 15.5%  
  - Avg CTAT → 15.3%  
  - Payment Method → 15%  
- **Decision Tree:**  
  - Ride Distance → 52%  
  - Avg VTAT → 26%  

📌 **Key Finding:**  
- **Ride Distance** (pricing) & **Wait Times** (customer experience) drive outcomes.  

---

## 📊 Streamlit Dashboard  

🔗 [**Live Demo App**](https://2aqyfhdehiughnvkzcvgb4.streamlit.app/)  

Features:  
- 📂 Upload CSV → auto-cleaning & preprocessing  
- 📊 Interactive dashboards (vehicle use, revenue, cancellations)  
- 🤖 Train/test ML models & compare performance  
- 📥 Download cleaned dataset  

---

## 🔑 Key Insights  

- ❌ **Cancellations cluster around specific pickup hubs**  
- 💰 **Revenue strongly tied to ride distance**  
- 🛵 **Autos & Bikes dominate short intra-city rides**  
- 🔮 Predictive ML models can forecast outcomes with **92%+ accuracy**  

---

## 📌 Strategic Recommendations  

### Operations  
- ⏱️ Reduce **VTAT & CTAT** → improve driver allocation.  
- 🚘 Enhance **driver supply** → cut down “No Driver Found”.  
- 📲 Improve **customer UX** → lower cancellation rate.  

### Revenue Growth  
- 💰 **Dynamic pricing** → distance + wait-time sensitive.  
- 🚕 **Fleet balance**: Autos + Go Mini (volume), Premier Sedan (premium).  
- 📅 **Seasonal planning** → optimize for peak demand months.  

### Technology  
- 🤖 **Predictive demand forecasting** → ML-powered scheduling.  
- 🛣️ **Route optimization** → reduce idle time.  
- 💳 **Payment integration** → streamline booking completions.  

---

## 🧠 Key Learnings  

- Handling **categorical + numerical preprocessing** in ML pipelines.  
- Avoiding **data leakage** in real-world datasets.  
- Comparing baseline vs ensemble models.  
- Deploying an **EDA + ML dashboard** in Streamlit.  

---

## 🚀 Deployment Guide  

Run Locally:
```bash
pip install -r requirements.txt
streamlit run app.py
