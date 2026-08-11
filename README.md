# App User Churn Prediction using K-Nearest Neighbors

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

## Overview

This project uses **K-Nearest Neighbors (KNN)** to predict whether an app user is likely to churn based on behavioral, usage, satisfaction, and account-related features.

The project focuses on a complete machine learning workflow, including data preprocessing, exploratory data analysis, categorical encoding, feature scaling, model training, K selection, and model evaluation.

## Problem Statement

Customer churn is an important problem for subscription-based applications and digital services. User behavior, satisfaction, service usage, payment patterns, and technical issues can influence whether a customer continues using a service.

The objective is to classify users into:

- `0` — Customer stays
- `1` — Customer churns

## Dataset

The dataset contains **3,520 records** and **20 columns**.

It includes realistic preprocessing challenges such as:

- Missing values
- Duplicate records
- Outliers
- Numerical features
- Categorical features
- Features with different scales
- Behavioral patterns related to churn

## Features

| Feature | Description |
|---|---|
| `Age` | Customer age |
| `Tenure_Months` | Duration of service usage |
| `Monthly_Charges` | Monthly service charges |
| `Total_Charges` | Total amount charged |
| `Support_Calls` | Number of support calls |
| `Data_Usage_GB` | Monthly data usage |
| `Num_Devices` | Number of connected devices |
| `Satisfaction_Score` | Customer satisfaction rating |
| `Payment_Delay_Days` | Average payment delay |
| `Monthly_App_Sessions` | Monthly application sessions |
| `Avg_Session_Duration_Min` | Average session duration |
| `Last_Login_Days_Ago` | Days since last login |
| `App_Crash_Count` | Number of application crashes |
| `Notification_Complaints` | Number of notification complaints |
| `Features_Used` | Number of features used |
| `Competitor_Usage_Hours` | Hours spent using competing services |
| `Contract_Length` | Contract duration |
| `Internet_Service` | Internet service type |
| `Payment_Method` | Payment method |
| `Churn` | Target variable |

## Machine Learning Workflow

```text
Data Loading
    |
Data Exploration
    |
Missing Value Analysis
    |
Duplicate Detection
    |
Outlier Detection
    |
Missing Value Handling
    |
Categorical Encoding
    |
Exploratory Data Analysis
    |
Train-Test Split
    |
Feature Scaling
    |
KNN Model Training
    |
K Selection
    |
Prediction
    |
Model Evaluation
Data Preprocessing
Missing Values

Missing numerical values are handled using median imputation, while categorical values are handled using the mode.

Duplicate Records

Duplicate records are identified and removed before model training.

Outlier Detection

Numerical features are analyzed using box plots to identify potential outliers.

Categorical Encoding

Categorical variables are converted into numerical features using one-hot encoding.

Feature Scaling

StandardScaler is applied because KNN relies on distance calculations between observations.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

The scaler is fitted only on the training data to prevent data leakage.

Model

The project uses KNeighborsClassifier from Scikit-learn.

Multiple K values are evaluated to identify a suitable value based on model performance.

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
Model Evaluation

The model is evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Classification Report

The project also compares different K values using Accuracy and F1 Score.

Visualizations

The following visualizations are included:

Churn Distribution
Monthly Charges vs Churn
Satisfaction Score vs Churn
Last Login vs Churn
K Value vs Model Performance
Confusion Matrix
Project Structure
app-user-churn-prediction/
│
├── data/
│   └── app_user_churn.csv
│
├── images/
│   ├── churn_distribution.png
│   ├── monthly_charges_vs_churn.png
│   ├── satisfaction_vs_churn.png
│   ├── last_login_vs_churn.png
│   ├── k_value_vs_model_performance.png
│   └── confusion_matrix.png
│
├── notebooks/
│   └── app_user_churn_prediction.ipynb
│
├── README.md
└── requirements.txt
Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook
Installation

Clone the repository:

git clone https://github.com/your-username/app-user-churn-prediction.git

Navigate to the project directory:

cd app-user-churn-prediction

Install the required dependencies:

pip install -r requirements.txt

Launch Jupyter Notebook:

jupyter notebook

Open the notebook from the notebooks directory.

Learning Outcomes

This project demonstrates:

Classification using KNN
Data preprocessing
Missing value handling
Duplicate detection
Outlier analysis
Categorical encoding
Feature scaling
Hyperparameter selection
Model evaluation
Exploratory data analysis
License

This project is licensed under the MIT License.

Author

Kaviyadharshini M

Computer Science Engineering
Artificial Intelligence | Machine Learning | Data Science

Support

If you found this project helpful, consider giving it a ⭐ to support my Machine Learning learning journey.