import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("GasProperties.csv")

print("Original data size:", data.shape)

# Z-score normalization + outlier removal
for col in data.columns:
    mean = data[col].mean()
    std = data[col].std()

    z = (data[col] - mean) / std

    # Remove outliers
    data = data[np.abs(z) <= 2]

print("After outlier removal:", data.shape)

# Normalize again (important after filtering)
data_norm = (data - data.mean()) / data.std()

# Features and target
X = data_norm.iloc[:, :-1]
y = data_norm.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# RMSE
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

print("\n--- NORMALIZED RESULTS ---")
print("Training RMSE:", rmse(y_train, model.predict(X_train)))
print("Testing RMSE:", rmse(y_test, model.predict(X_test)))