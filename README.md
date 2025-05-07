# Taxi Fare Prediction

This project aims to build a linear regression model to predict the final taxi fare amount based on trip-related features, using real-world data from the City of Chicago Open Data portal.

---

## 🚀 Project Summary

The dataset includes detailed taxi trip information such as trip duration, distance, fare, tips, tolls, extras, and timestamps. The target variable is `Trip Total`, and the goal is to predict it as accurately as possible using machine learning techniques.

We used Python and the Scikit-Learn library to:

* Clean and preprocess the dataset (handling nulls, filtering outliers)
* Engineer relevant features
* Train a Linear Regression model
* Evaluate it using standard metrics (MAE, RMSE, MSE)

---

## 📁 Project Structure

```bash
.
├── data
│   └── Taxi_Trips_2024_clean.csv      # Cleaned dataset
├── notebooks
│   └── taxi_fare_prediction_final.ipynb  # Jupyter notebook for the analysis
├── src
│   └── utils.py                        # (Optional) helper functions if modularized
├── requirements.txt                   # List of dependencies
└── README.md                          # Project documentation

📊 Dataset Overview
Source: Chicago Data Portal - Taxi Trips

Original Columns:

Trip ID, Taxi ID, Start/End Timestamps, Trip Miles/Seconds, Fare, Tips, Tolls, etc.
Final Selected Features:

Trip Seconds
Trip Miles
Fare
Target:

Trip Total
🧹 Data Cleaning
Removed rows with null or zero/negative values in core columns.
Converted timestamp columns to datetime.
Removed outliers using the IQR method.
Resulting dataset size:

From 711,775 rows → 507,454 rows
🤖 Model & Evaluation
Model: Linear Regression

Final coefficients:

Feature	Coefficient
Trip Seconds	≈ -0.00
Trip Miles	≈ 0.19
Fare	≈ 1.15
Intercept: ≈ 0.48

After outlier removal:

MAE: 1.79
RMSE: 2.23
MSE: 4.99
This shows a significant improvement in accuracy by cleaning data and filtering out outliers.

⚙️ Installation & Setup
Clone this repo:
bash

Copy
git clone https://github.com/mo-qanbari/taxi-fare-prediction.git
Install dependencies:
bash

Copy
pip install -r requirements.txt
Run the notebook:
bash

Copy
cd notebooks
jupyter notebook taxi_fare_prediction_final.ipynb
🛠️ Tools & Libraries
Python 3.x
Pandas, NumPy, Matplotlib, Seaborn
Scikit-learn
📌 License
This project is open-source and available under the MIT License.

✍️ Author
Mohammad Reza Molla Ghanbari Kashi
