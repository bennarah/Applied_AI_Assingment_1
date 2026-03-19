import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = pd.read_csv("GasProperties.csv")

print("Columns:", data.columns)

# Features (all except last column)
X = data.iloc[:, :-1]

# Target (last column)
y = data.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Results
print("\n--- RESULTS ---")
print("Training RMSE:", rmse(y_train, model.predict(X_train)))
print("Testing RMSE:", rmse(y_test, model.predict(X_test)))