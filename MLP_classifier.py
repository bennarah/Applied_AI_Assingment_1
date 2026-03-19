from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load data
data = load_iris()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLP model (as required)
model = MLPClassifier(
    hidden_layer_sizes=(3,),  # ONE hidden layer with 3 neurons
    activation='relu',
    learning_rate_init=0.01,
    max_iter=1000
)

# Train
model.fit(X_train, y_train)

# Results
print("\n--- MLP RESULTS ---")
print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", model.score(X_test, y_test))