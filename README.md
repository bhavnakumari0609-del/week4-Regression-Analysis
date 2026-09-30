# Week 4 - Predictive Modeling and Regression Analysis

## Project Overview

This project demonstrates how predictive modeling and regression analysis can be used to forecast future business trends using Python. The project uses business-related data to study the relationship between factors such as advertising expenditure, number of employees, website visits, and customer count with overall sales.

The project covers the complete regression workflow, including dataset creation, data preparation, model training, prediction, evaluation, cross-validation, and visualization.

## Objective

The main objectives of this project are:

- To understand predictive analytics for business decision-making.
- To prepare business data for regression analysis.
- To build different regression models using Python.
- To compare model performance using evaluation metrics.
- To apply cross-validation for reliable model evaluation.
- To visualize actual and predicted business sales.

## Dataset

The dataset contains 1,000 business records with the following variables:

| Feature | Description |
|---|---|
| Advertising_Spend | Amount spent on advertising |
| Employees | Number of employees |
| Website_Visits | Number of website visits |
| Customer_Count | Number of customers |
| Sales | Target business sales |

The dataset was generated using Python and saved as:

`business_regression_data.csv`

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Regression Models

Three regression approaches were implemented:

### 1. Linear Regression

Linear Regression was used as the baseline model to estimate the relationship between the input business variables and sales.

### 2. Ridge Regression

Ridge Regression was included to reduce the effect of multicollinearity and improve model stability. Feature scaling was applied using `StandardScaler`.

### 3. Polynomial Regression

Polynomial Regression was used to capture possible non-linear relationships between the business features and sales.

## Data Pre-processing

The dataset was loaded using Pandas and divided into independent features and the target variable. The data was then divided into training and testing sets using `train_test_split`.

For Ridge and Polynomial Regression, feature scaling was performed using `StandardScaler`. Polynomial features were generated using `PolynomialFeatures`.

## Model Training

The models were trained using Scikit-learn. Predictions were generated using the test dataset.

The following functions and classes were used:

- `pd.read_csv()`
- `train_test_split()`
- `LinearRegression()`
- `Ridge()`
- `PolynomialFeatures()`
- `StandardScaler()`
- `Pipeline()`

## Model Validation

Five-fold cross-validation was used to evaluate model stability. RMSE was used as the cross-validation scoring metric.

The project also evaluates models using:

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

Lower RMSE and MAE indicate smaller prediction errors, while a higher R² indicates that the model explains a larger proportion of the variation in the target variable.

## Visualization

Two visualizations were created:

1. `actual_vs_predicted_sales.png`  
   Shows the relationship between actual and predicted sales values.

2. `model_performance_comparison.png`  
   Compares the RMSE values of the regression models.

## Project Files

```text
Week4-Regression-Analysis/
│
├── create_dataset.py
├── predictive_model.py
├── visualize_results.py
├── business_regression_data.csv
├── regression_model_results.csv
├── actual_vs_predicted_sales.png
├── model_performance_comparison.png
└── README.md