import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Convert to binary classification (class 0 vs others)
y = np.where(y == 0, 0, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize parameters
w = np.zeros(X.shape[1])   # weights
b = 0                      # bias
learning_rate = 0.01
epochs = 100

# Activation function
def step(x):
    return 1 if x >= 0 else 0

# Training loop
for epoch in range(epochs):
    for i in range(len(X_train)):
        linear_output = np.dot(w, X_train[i]) + b
        y_pred = step(linear_output)

        error = y_train[i] - y_pred

        # Update rule
        w += learning_rate * error * X_train[i]
        b += learning_rate * error

# Accuracy function
def accuracy(X, y):
    correct = 0
    for i in range(len(X)):
        y_pred = step(np.dot(w, X[i]) + b)
        if y_pred == y[i]:
            correct += 1
    return correct / len(X)

# Results
print("\n--- PERCEPTRON RESULTS ---")
print("Weights:", w)
print("Bias:", b)
print("Learning rate:", learning_rate)
print("Epochs:", epochs)

print("\nTraining Accuracy:", accuracy(X_train, y_train))
print("Testing Accuracy:", accuracy(X_test, y_test))