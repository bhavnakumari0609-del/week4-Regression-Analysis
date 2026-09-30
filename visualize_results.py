import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv("business_regression_data.csv")

# Features and target
X = df[
    [
        "Advertising_Spend",
        "Employees",
        "Website_Visits",
        "Customer_Count"
    ]
]

y = df["Sales"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Linear Regression RMSE: {rmse:.2f}")

# -------------------------------
# Chart 1: Actual vs Predicted
# -------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions, alpha=0.6)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

# Reference line
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()
plt.savefig("actual_vs_predicted_sales.png", dpi=300)
plt.show()


# --------------------------------
# Chart 2: Model Performance
# --------------------------------

results = pd.read_csv("regression_model_results.csv")

plt.figure(figsize=(9, 6))

plt.bar(
    results["Model"],
    results["RMSE"]
)

plt.xlabel("Regression Model")
plt.ylabel("RMSE")
plt.title("Regression Model Performance Comparison")

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("model_performance_comparison.png", dpi=300)
plt.show()

print("\nCharts created successfully!")