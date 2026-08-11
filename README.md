# App User Churn Prediction using K-Nearest Neighbors

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

## Overview

This project implements a **K-Nearest Neighbors (KNN)** classification model to predict whether an app user is likely to stop using a service.

The project focuses on understanding a complete machine learning workflow, including data exploration, preprocessing, missing value handling, duplicate detection, outlier analysis, categorical encoding, feature scaling, K selection, model training, and classification evaluation.

---

## Problem Statement

User churn is an important business problem for subscription-based applications and digital services.

Users may stop using a service because of factors such as:

- Low satisfaction
- Infrequent app usage
- Long periods since the last login
- Frequent application crashes
- High monthly charges
- Frequent support issues
- Payment delays
- Excessive use of competing services

The objective of this project is to use user behavior and account information to predict whether a customer will **stay** or **churn**.

---

## Target Variable

The target variable is:

```text
Churn
````

| Value | Meaning         |
| ----: | --------------- |
|   `0` | Customer stays  |
|   `1` | Customer churns |

---

## Dataset

The dataset contains **3,520 records** and **20 columns**.

It was designed to simulate realistic app-user behavior and intentionally contains data quality issues for preprocessing practice.

The dataset includes:

* Missing values
* Duplicate records
* Numerical features
* Categorical features
* Different feature scales
* Intentional outliers
* Behavioral patterns related to churn

---

## Features

| Feature                    | Description                               |
| -------------------------- | ----------------------------------------- |
| `Age`                      | Age of the customer                       |
| `Tenure_Months`            | Number of months using the service        |
| `Monthly_Charges`          | Monthly service charges                   |
| `Total_Charges`            | Total amount charged                      |
| `Support_Calls`            | Number of customer support calls          |
| `Data_Usage_GB`            | Monthly data usage                        |
| `Num_Devices`              | Number of connected devices               |
| `Satisfaction_Score`       | Customer satisfaction rating              |
| `Payment_Delay_Days`       | Average payment delay                     |
| `Monthly_App_Sessions`     | Number of app sessions per month          |
| `Avg_Session_Duration_Min` | Average session duration                  |
| `Last_Login_Days_Ago`      | Days since the customer's last login      |
| `App_Crash_Count`          | Number of application crashes             |
| `Notification_Complaints`  | Number of notification-related complaints |
| `Features_Used`            | Number of service features used           |
| `Competitor_Usage_Hours`   | Hours spent using competing services      |
| `Contract_Length`          | Customer contract duration                |
| `Internet_Service`         | Type of internet service                  |
| `Payment_Method`           | Customer payment method                   |
| `Churn`                    | Target variable                           |

---

## Why K-Nearest Neighbors?

KNN is a distance-based supervised learning algorithm.

Instead of learning an explicit mathematical equation, KNN classifies a new observation based on the classes of its nearest observations.

```text
Training Data
     |
     v
Calculate Distance
     |
     v
Find K Nearest Neighbors
     |
     v
Majority Voting
     |
     v
Predicted Class
```

Feature scaling is particularly important for KNN because features with larger numerical ranges can otherwise dominate the distance calculation.

---

## Machine Learning Workflow

```text
Data Loading
     |
     v
Data Exploration
     |
     v
Missing Value Analysis
     |
     v
Duplicate Detection
     |
     v
Outlier Detection
     |
     v
Missing Value Handling
     |
     v
Categorical Encoding
     |
     v
Exploratory Data Analysis
     |
     v
Feature-Target Separation
     |
     v
Train-Test Split
     |
     v
Feature Scaling
     |
     v
KNN Model
     |
     v
K Selection
     |
     v
Prediction
     |
     v
Model Evaluation
```

---

## Data Preprocessing

### Missing Values

Missing values are identified using:

```python
df.isnull().sum()
```

Missing values in numerical columns are handled using median imputation:

```python
numeric_columns = df.select_dtypes(include=np.number).columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)
```

Categorical missing values are handled using the mode:

```python
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )
```

### Duplicate Records

Duplicate records are checked using:

```python
df.duplicated().sum()
```

Duplicates are removed before model training:

```python
df.drop_duplicates(inplace=True)
```

### Outlier Detection

Numerical features are examined using box plots to identify potential outliers.

Outliers are analyzed before deciding whether they should be removed or retained.

### Categorical Encoding

Categorical features are converted into numerical representations using one-hot encoding:

```python
df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)
```

---

## Exploratory Data Analysis

The project includes visualizations to understand the data and identify relationships with churn.

### Churn Distribution

Shows the number of users who stayed and churned.

### Monthly Charges vs Churn

Examines whether monthly charges differ between churned and retained users.

### Satisfaction Score vs Churn

Analyzes the relationship between customer satisfaction and churn.

### Last Login vs Churn

Examines whether users who have not logged in recently are more likely to churn.

### Correlation Analysis

A correlation heatmap is used to analyze relationships between numerical variables.

---

## Feature and Target Separation

```python
X = df.drop("Churn", axis=1)
y = df["Churn"]
```

---

## Train-Test Split

The dataset is divided into training and testing sets using an 80:20 split.

Stratification is used to preserve the class distribution.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## Feature Scaling

Standardization is applied because KNN relies on distance calculations.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data to prevent data leakage.

---

## KNN Model

The KNN classifier is implemented using Scikit-learn:

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
```

---

## Selecting the Best K

Instead of selecting K arbitrarily, multiple K values are tested.

```python
k_values = range(1, 21)

accuracy_scores = []
f1_scores = []

for k in k_values:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(X_train_scaled, y_train)

    y_pred_k = knn.predict(X_test_scaled)

    accuracy_scores.append(
        accuracy_score(y_test, y_pred_k)
    )

    f1_scores.append(
        f1_score(y_test, y_pred_k)
    )
```

The results are visualized using a K value versus model performance plot.

The best K is selected based on model performance rather than using a fixed value.

---

## Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
```

---

## Confusion Matrix

The confusion matrix provides a detailed view of correct and incorrect classifications.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stay", "Churn"],
    yticklabels=["Stay", "Churn"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")

plt.show()
```

---

## Visualizations

The following visualizations are included in the project:

* Churn Distribution
* Monthly Charges vs Churn
* Satisfaction Score vs Churn
* Last Login vs Churn
* K Value vs Model Performance
* Confusion Matrix

---

## Project Structure

```text
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
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/app-user-churn-prediction.git
```

### Navigate to the Project

```bash
cd app-user-churn-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook from the `notebooks` directory.

---

## Learning Outcomes

This project covers the following machine learning concepts:

* Classification
* K-Nearest Neighbors
* Missing Value Handling
* Duplicate Detection
* Outlier Detection
* Categorical Encoding
* Feature Scaling
* Train-Test Splitting
* Data Leakage Prevention
* Hyperparameter Selection
* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Exploratory Data Analysis

---

## Future Improvements

* Compare KNN with Logistic Regression
* Compare KNN with Decision Tree
* Compare KNN with Random Forest
* Perform cross-validation
* Perform systematic hyperparameter tuning
* Address class imbalance
* Experiment with different distance metrics
* Apply feature selection
* Deploy the model using Streamlit

---

## License

This project is licensed under the MIT License.

---

## Author

**Kaviyadharshini M**

Computer Science Engineering
Artificial Intelligence | Machine Learning | Data Science

---

## Support

If you found this project helpful, consider giving it a ⭐ to support my Machine Learning learning journey.

```
```
