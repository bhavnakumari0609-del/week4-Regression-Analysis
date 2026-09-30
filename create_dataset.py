import pandas as pd
import numpy as np

np.random.seed(42)

# Number of business records
n = 1000

# Generate business data
advertising_spend = np.random.uniform(5000, 100000, n)
employees = np.random.randint(5, 101, n)
website_visits = np.random.randint(1000, 50000, n)
customer_count = np.random.randint(100, 10000, n)

# Generate sales using a realistic business relationship
sales = (
    50000
    + 2.8 * advertising_spend
    + 15000 * employees
    + 8 * website_visits
    + 120 * customer_count
    + np.random.normal(0, 150000, n)
)

# Create DataFrame
df = pd.DataFrame({
    "Advertising_Spend": advertising_spend,
    "Employees": employees,
    "Website_Visits": website_visits,
    "Customer_Count": customer_count,
    "Sales": sales
})

# Save dataset
df.to_csv("business_regression_data.csv", index=False)

print("Dataset created successfully!")
print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nStatistical Summary:")
print(df.describe())