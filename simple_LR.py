import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = pd.read_csv("study_data.csv")

print("First 5 rows:")
print(data.head())

# Features and target
X = data[['Hours']]
y = data['Score']

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining size:", len(X_train))
print("Testing size:", len(X_test))

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Print results
print("\n--- RESULTS ---")
print("Training RMSE:", rmse(y_train, y_train_pred))
print("Testing RMSE:", rmse(y_test, y_test_pred))

# Print learned equation
w0 = model.intercept_
w1 = model.coef_[0]

print("\n--- MODEL EQUATION ---")
print(f"y = {w0:.2f} + {w1:.2f} * x")