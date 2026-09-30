import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 1. Load dataset
df = pd.read_csv("business_regression_data.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Separate features and target
X = df[
    [
        "Advertising_Spend",
        "Employees",
        "Website_Visits",
        "Customer_Count"
    ]
]

y = df["Sales"]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 4. Define models
models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ]),

    "Polynomial Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("scaler", StandardScaler()),
        ("linear", LinearRegression())
    ])
}


# 5. Train and evaluate models
results = []

print("\n========== MODEL PERFORMANCE ==========")

for name, model in models.items():

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    print(f"\n{name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")


# 6. Cross-validation
print("\n========== CROSS-VALIDATION ==========")

for name, model in models.items():

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    mean_cv_rmse = -cv_scores.mean()

    print(f"{name}: Mean CV RMSE = {mean_cv_rmse:.2f}")


# 7. Compare models
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# 8. Save results
results_df.to_csv(
    "regression_model_results.csv",
    index=False
)

print("\nModel results saved to regression_model_results.csv")