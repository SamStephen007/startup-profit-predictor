import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv("50_Startups.csv")

# Check the first few rows
print(df.head())

# Separate categorical and numerical data
X = df.drop('Profit', axis=1)
y = df['Profit']

# One-hot encode the 'State' column
X = pd.get_dummies(X, columns=['State'], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# lr = LinearRegression()
# lr.fit(X_train, y_train)
# y_pred_lr = lr.predict(X_test)


ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)


# print("=== Linear Regression ===")
# print("MSE:", mean_squared_error(y_test, y_pred_lr))
# print("R²:", r2_score(y_test, y_pred_lr))

print("\n=== Ridge Regression ===")
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("R²:", r2_score(y_test, y_pred_ridge))


plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Profit', marker='o')
# plt.plot(y_pred_lr, label='Linear Predicted', marker='x')
plt.plot(y_pred_ridge, label='Ridge Predicted', marker='s')
plt.legend()
plt.title("Actual vs Predicted Profits")
plt.xlabel("Test Sample Index")
plt.ylabel("Profit")
plt.grid(True)
plt.show()
import pickle
import os

# Create model directory
os.makedirs("model", exist_ok=True)

# Save the trained model and feature names
pickle.dump(ridge, open("model/ridge_model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("model/features.pkl", "wb"))
